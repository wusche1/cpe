import heapq
import json
import math
import os
import random
from collections import defaultdict

from lib.jev import judge_many as jev_many
from lib.judge import judge_many as ask_many


def main(log_path, completions_path, views, top_k, judge_k, random_k, seed, control, reader, reader_template,
         reader_repeats, discriminate_template, discriminate_repeats, judge_model, judge_concurrency, text_questions,
         rubric_questions):
    os.makedirs(log_path, exist_ok=True)
    rng = random.Random(seed)
    scores = defaultdict(list)
    for n, l in enumerate(open(completions_path)):
        r = json.loads(l)
        if r["nll_ft"] is not None:
            s = {"diff": math.exp(r["nll_base"]) - math.exp(r["nll_ft"]), "ratio": r["nll_base"] - r["nll_ft"]}
            for view, v in views.items():
                if f"{r['corpus']}|{r['mode']}" in v["configs"]:
                    scores[view, r["model"]].append((s[v["score"]], n))
    picks = {}
    for (view, model), xs in scores.items():
        for rank, (s, n) in enumerate(heapq.nlargest(judge_k, xs)):
            picks[view, model, "top", rank] = n
        for rank, (s, n) in enumerate(rng.sample(xs, random_k)):
            picks[view, model, "random", rank] = n
    wanted = set(picks.values())
    rows = {n: json.loads(l) for n, l in enumerate(open(completions_path)) if n in wanted}
    ans = jev_many({n: {"text": r["prefill"] + r["text"]} for n, r in rows.items()}, text_questions, judge_model,
                   judge_concurrency)
    sets = defaultdict(list)
    for (view, model, kind, rank), n in sorted(picks.items()):
        sets[view, model, kind].append({**rows[n], "rank": rank, **{q: ans[n][q].noul for q in text_questions}})
    with open(os.path.join(log_path, "sets.json"), "w") as f:
        json.dump({"|".join(k): v for k, v in sets.items()}, f, indent=1)

    models = sorted({m for _, m in scores})
    listing = lambda view, model: "\n".join(f"{r['rank'] + 1}. {r['prefill']} | {r['text']}" for r in sets[view, model, "top"][:top_k])
    desc = ask_many({f"{view}|{m}|{i}": reader_template.format(k=top_k, completions=listing(view, m))
                     for view in views for m in models for i in range(reader_repeats)}, **reader)
    grades = jev_many({k: {"description": json.dumps(d)} for k, d in desc.items()}, rubric_questions, judge_model,
                      judge_concurrency)
    orders = {f"{view}|{i}": rng.sample(models, len(models)) for view in views for i in range(discriminate_repeats)}
    disc = ask_many({k: discriminate_template.format(k=top_k, sets="\n\n".join(
        f"=== MODEL {chr(65 + j)} ===\n{listing(k.split('|')[0], m)}" for j, m in enumerate(o))) for k, o in orders.items()}, **reader)
    with open(os.path.join(log_path, "reader.json"), "w") as f:
        json.dump({"descriptions": {k: {**d, **{q: grades[k][q].noul for q in rubric_questions}} for k, d in desc.items()},
                   "discrimination": {k: {"order": o, "answer": disc[k], "correct": o[ord(disc[k]["control"]) - 65] == control}
                                      for k, o in orders.items()}}, f, indent=1)

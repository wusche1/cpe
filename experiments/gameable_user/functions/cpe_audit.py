import json
import os
import subprocess
import sys
import time
from collections import defaultdict

from lib.compose import compose_adapters
from lib.generation import build_prompts, generate_in_subprocess
from lib.grpo import prepare_factors
from lib.jev import judge_many

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _judge(states, questions, model, concurrency, tries):
    out, keys = {}, list(states)
    for i in range(0, len(keys), 512):
        chunk = {k: states[k] for k in keys[i:i + 512]}
        for attempt in range(tries):
            try:
                out.update(judge_many(chunk, questions, model, concurrency))
                break
            except Exception:
                if attempt == tries - 1:
                    raise
                time.sleep(2 ** attempt)
    return out


def main(log_path, model_name, system_prompt, enable_thinking, sources, train_path, heldout_path, num_train,
         num_score, num_confirm, kinds, control_prompts, max_seq_len, cpe, generation, factor_scales, num_random,
         random_seed, top_k, random_k, min_coherent, confirm_samples, baseline_reps, judge_model, judge_questions,
         control_questions, judge_threshold, judge_concurrency, jev_tries, score_unsteered_samples, pick_kind):
    from transformers import AutoTokenizer

    os.makedirs(log_path, exist_ok=True)
    work = os.path.abspath(os.path.join("tmp", os.path.basename(os.path.normpath(log_path))))
    tok = AutoTokenizer.from_pretrained(model_name, truncation_side="left")
    sources = {k: v and os.path.abspath(v) for k, v in sources.items()}
    train = [json.loads(l) for l in open(train_path)][:num_train]
    held = [json.loads(l) for l in open(heldout_path)]
    score_s, confirm_s = held[:num_score], held[num_score:num_score + num_confirm]
    chat = lambda msgs: build_prompts(tok, msgs, system_prompt, enable_thinking)
    timings = {}

    def generate(tag, adapters, meta):
        t = time.time()
        comps = generate_in_subprocess(work, tag, model_name=model_name, adapters=adapters,
                                       prompts=chat([m["message"] for m in meta]), **generation)
        timings[tag] = {"seconds": time.time() - t, "completions": len(adapters) * len(meta)}
        rows = [{"arm": a, **m, "text": c} for a, cs in comps.items() for m, c in zip(meta, cs)]
        for qs, sel in ((judge_questions, lambda r: r["kind"] != "control"), (control_questions, lambda r: r["kind"] == "control")):
            idx = [i for i, r in enumerate(rows) if sel(r)]
            ans = _judge({i: {"user_message": rows[i]["message"], "assistant_response": rows[i]["text"] or "(empty)"}
                          for i in idx}, qs, judge_model, judge_concurrency, jev_tries)
            for i in idx:
                rows[i].update({q: ans[i][q].noul for q in qs})
        for r in rows:
            r["hit"] = r["urges" if r["kind"] != "control" else "complied"] > judge_threshold and r["coherent"] > judge_threshold
        with open(os.path.join(log_path, f"{tag}.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        return rows

    def rates(rows, key):
        by = defaultdict(lambda: defaultdict(list))
        for r in rows:
            by[key(r)][r["kind"]].append(r)
        return {a: {k: {"n": len(v), "hits": sum(r["hit"] for r in v),
                        **{q: sum(r[q] for r in v) / len(v) for q in ("coherent", "anomalous") if q in v[0]}}
                    for k, v in ks.items()} for a, ks in by.items()}

    token_ids = [tok.encode(p, truncation=True, max_length=max_seq_len, add_special_tokens=False)
                 for p in chat([s["neutral"] for s in train])]
    raw = {}
    for src, adapter in sources.items():
        root = os.path.abspath(os.path.join(log_path, "factors", src))
        args_path = os.path.join(os.path.abspath(log_path), f"train_args_{src}.json")
        with open(args_path, "w") as f:
            json.dump(dict(model_name=model_name, sae_config=None, token_ids=token_ids, adapter_root=root, labeled=None,
                           sft_config=None, steer_config=None, base_adapter=adapter,
                           steer_path=os.path.join(work, "steer.pt"),
                           log_dir=os.path.abspath(os.path.join(log_path, "training", src)), **cpe), f)
        t = time.time()
        subprocess.run([sys.executable, "-m", "lib.train_proc", args_path], check=True, cwd=REPO,
                       env=dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", "")))
        timings[f"cpe_train_{src}"] = {"seconds": time.time() - t}
        raw[src] = [os.path.join(root, n) for n in sorted(os.listdir(root), key=lambda n: int(n.split("_")[1]))]

    scaled = {}
    for src in sources:
        for s in factor_scales:
            for kind, seed, dirs in (("cpe", None, raw[src]), ("random", random_seed, raw[src][:num_random])):
                for i, d in enumerate(prepare_factors(dirs, s, seed, os.path.join(work, f"{src}_{kind}_s{s}"))):
                    scaled[f"{src}|{kind}{i}_s{s}"] = d

    def on(target, name):
        if not sources[target]:
            return scaled[name]
        return compose_adapters(sources[target], scaled[name], os.path.join(work, "composed", target, name.replace("|", "_")))

    meta = [{"kind": k, "scenario_id": s["scenario_id"], "message": s[k]} for s in score_s for k in kinds]
    stats = rates(generate("score", {n: on(n.split("|")[0], n) for n in scaled}, meta), lambda r: r["arm"])
    picks = {}
    for src in sources:
        names = [n for n in stats if n.startswith(f"{src}|cpe")]
        rand = [n for n in stats if n.startswith(f"{src}|random")]
        picks[src] = {
            "oracle": sorted(names, key=lambda n: -stats[n][pick_kind]["hits"])[:top_k],
            "agnostic": sorted([n for n in names if stats[n][pick_kind]["coherent"] >= min_coherent],
                               key=lambda n: -stats[n][pick_kind]["anomalous"])[:top_k],
            "random": sorted(rand, key=lambda n: -stats[n][pick_kind]["hits"])[:random_k]}
    summary = {"score": stats, "picks": picks, "timings": timings}
    if score_unsteered_samples:
        summary["score_unsteered"] = rates(generate("score_unsteered", {f"{src}|none": sources[src] for src in sources},
                                                    meta * score_unsteered_samples), lambda r: r["arm"])
    with open(os.path.join(log_path, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    meta = [{"kind": k, "scenario_id": s["scenario_id"], "message": s[k]}
            for s in confirm_s for k in kinds for _ in range(confirm_samples)]
    meta += [{"kind": "control", "scenario_id": i, "message": p}
             for i, p in enumerate(control_prompts) for _ in range(confirm_samples)]
    arms = {}
    for target in sources:
        for j in range(baseline_reps):
            arms[f"{target}|none|baseline|{j}"] = sources[target]
        for src, groups in picks.items():
            for group, names in groups.items():
                for n in names:
                    arms[f"{target}|{src}|{group}|{n.split('|')[1]}"] = on(target, n)
    rows = generate("confirm", arms, meta)
    summary["confirm"] = rates(rows, lambda r: "|".join(r["arm"].split("|")[:3]))
    summary["confirm_by_arm"] = rates(rows, lambda r: r["arm"])
    with open(os.path.join(log_path, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["confirm"], indent=2))

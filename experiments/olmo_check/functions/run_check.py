import json
import os
import subprocess
import sys
import time
from collections import defaultdict

from lib.generation import build_prompts, generate_completions
from lib.grpo import prepare_factors
from lib.jev import judge_many

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main(log_path, model_name, situations, prompt_template, conditions, num_train, num_eval, system_prompt,
         max_seq_len, cpe, generation, factor_scales, num_random, random_seed, baseline_samples, top_k,
         confirm_samples, min_coherent, judge_model, judge_questions, judge_threshold, judge_concurrency):
    from transformers import AutoTokenizer

    os.makedirs(log_path, exist_ok=True)
    work = os.path.abspath(os.path.join("tmp", os.path.basename(os.path.normpath(log_path))))
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left")
    train, ev, conf = situations[:num_train], situations[num_train:num_train + num_eval], situations[num_train + num_eval:]
    request = lambda s, c: prompt_template.format(situation=s) + conditions[c]
    timings = {}

    def run(adapters, sits, n, stage):
        meta = [(s, c) for s in sits for c in conditions for _ in range(n)]
        prompts = build_prompts(tokenizer, [request(s, c) for s, c in meta], system_prompt)
        t = time.time()
        comps = generate_completions(model_name, adapters, prompts, **generation)
        timings[stage] = {"generate_seconds": time.time() - t, "completions": len(adapters) * len(meta)}
        rows = [{"stage": stage, "factor": name, "situation": s, "condition": c, "completion": comp}
                for name, cs in comps.items() for (s, c), comp in zip(meta, cs)]
        answers = judge_many({i: {"request": request(r["situation"], r["condition"]),
                                  "response": r["completion"] or "(empty)"} for i, r in enumerate(rows)},
                             judge_questions, judge_model, judge_concurrency)
        for i, r in enumerate(rows):
            r.update({q: answers[i][q].noul for q in judge_questions})
            r["hit"] = r["harmful"] > judge_threshold and r["coherent"] > judge_threshold
        with open(os.path.join(log_path, "completions.jsonl"), "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return rows

    def rates(rows):
        by = defaultdict(lambda: defaultdict(list))
        for r in rows:
            by[r["factor"]][r["condition"]].append(r)
            by[r["factor"]]["all"].append(r)
        return {f: {c: {"n": len(v), "hits": sum(r["hit"] for r in v),
                        **{q: sum(r[q] for r in v) / len(v) for q in judge_questions}}
                    for c, v in cs.items()} for f, cs in by.items()}

    def save(summary):
        with open(os.path.join(log_path, "summary.json"), "w") as f:
            json.dump({**summary, "timings": timings}, f, indent=2)

    adapter_root = os.path.join(work, "adapters")
    token_ids = [tokenizer.encode(p, truncation=True, max_length=max_seq_len, add_special_tokens=False)
                 for p in build_prompts(tokenizer, [request(s, c) for s in train for c in cpe["train_conditions"]],
                                        system_prompt)]
    args_path = os.path.join(os.path.abspath(log_path), "train_args.json")
    with open(args_path, "w") as f:
        json.dump(dict(model_name=model_name, sae_config=None, token_ids=token_ids, adapter_root=adapter_root,
                       labeled=None, sft_config=None, steer_config=None, base_adapter=None,
                       steer_path=os.path.join(work, "steer.pt"),
                       log_dir=os.path.abspath(os.path.join(log_path, "training")),
                       **{k: v for k, v in cpe.items() if k != "train_conditions"}), f)
    t = time.time()
    subprocess.run([sys.executable, "-m", "lib.train_proc", args_path], check=True, cwd=REPO,
                   env=dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", "")))
    timings["cpe_train"] = {"seconds": time.time() - t}

    summary = {"baseline": rates(run({"baseline": None}, ev + conf, baseline_samples, "baseline"))["baseline"]}
    save(summary)

    raw = [os.path.join(adapter_root, n) for n in sorted(os.listdir(adapter_root), key=lambda n: int(n.split("_")[1]))]
    adapters = {}
    for s in factor_scales:
        for kind, seed, dirs in (("cpe", None, raw), ("random", random_seed, raw[:num_random])):
            for i, d in enumerate(prepare_factors(dirs, s, seed, os.path.join(work, f"{kind}_s{s}"))):
                adapters[f"{kind}{i}_s{s}"] = d

    stats = rates(run(adapters, ev, 1, "factors"))
    cpe_names = [f for f in stats if f.startswith("cpe")]
    oracle = sorted(cpe_names, key=lambda f: -stats[f]["all"]["hits"])[:top_k]
    unsup = sorted([f for f in cpe_names if stats[f]["all"]["coherent"] >= min_coherent],
                   key=lambda f: -stats[f]["all"]["anomalous"])[:top_k]
    rand = sorted([f for f in stats if f.startswith("random")], key=lambda f: -stats[f]["all"]["hits"])[:top_k]
    summary.update(factors=stats, oracle=oracle, unsup=unsup, random_top=rand)
    save(summary)

    arms = {"baseline": None, **{f: adapters[f] for f in dict.fromkeys(oracle + unsup + rand)}}
    summary["confirm"] = rates(run(arms, conf, confirm_samples, "confirm"))
    save(summary)
    print(json.dumps({"baseline": summary["baseline"], "confirm": summary["confirm"]}, indent=2))

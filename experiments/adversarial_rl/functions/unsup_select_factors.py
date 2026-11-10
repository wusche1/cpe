import json
import os
import sys
from collections import defaultdict

import yaml

from lib.cpe.factors import FactorSet

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vuln_judge import flag


def main(log_path: str, cpe_run_dir: str, tasks_path: str, model_name: str, num_prompts: int, min_pass_ratio: float,
         top_k: int, judge_config: str, out_dir: str):
    os.makedirs(log_path, exist_ok=True)
    tasks = {t["task_id"]: t for t in map(json.loads, open(tasks_path))}
    rows = [json.loads(l) for l in open(os.path.join(cpe_run_dir, "completions.jsonl"))]
    first = sorted({r["prompt_idx"] for r in rows if r["stage"] == "halving"})[:num_prompts]
    rows = [r for r in rows if r["stage"] in ("halving", "baseline") and r["prompt_idx"] in first]
    judged = flag([(tasks[r["task_id"]]["prompt"], r["completion"]) for r in rows], yaml.safe_load(open(judge_config)))
    by = defaultdict(list)
    for r, j in zip(rows, judged):
        by[r["factor"]].append({"passed": r["passed"], "sus": j["p_vuln"], "payload": r["jev"]["vulnerable"]})
    mean = {f: {k: sum(float(x[k]) for x in v) / len(v) for k in ("passed", "sus", "payload")} for f, v in by.items()}
    base = mean.pop("baseline")
    coherent = [f for f in mean if mean[f]["passed"] >= min_pass_ratio * base["passed"]]
    chosen = sorted(coherent, key=lambda f: (-mean[f]["sus"], f))[:top_k]
    fs = FactorSet.load(os.path.join(cpe_run_dir, "training", "factors"))
    dirs = [fs.to_peft(int(f.split("_")[1]), os.path.join(out_dir, f"factor_{j}"), model_name)
            for j, f in enumerate(chosen)]
    json.dump({"baseline": base, "num_coherent": len(coherent), "chosen": chosen, "dirs": dirs,
               "factors": mean}, open(os.path.join(log_path, "unsup_selection.json"), "w"), indent=2)
    print("baseline", base, "coherent", len(coherent), flush=True)
    print("chosen (payload rate is evaluation only):", [(f, round(mean[f]["sus"], 2), mean[f]["payload"]) for f in chosen],
          flush=True)

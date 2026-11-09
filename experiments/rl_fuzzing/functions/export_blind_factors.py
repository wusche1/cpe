import json
import os
from collections import defaultdict

from lib.cpe.factors import FactorSet


def main(log_path: str, cpe_run_dir: str, model_name: str, num_prompts: int, top_k: int, out_dir: str,
         exclude_detected: bool):
    """Detector-free selection: the top_k factors of a CPE population with the lowest test pass rate on the first
    num_prompts halving prompts (most behaviour-changing), ties by factor index. Exported unit-norm (scale in RL)."""
    os.makedirs(log_path, exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(cpe_run_dir, "completions.jsonl"))]
    rows = [r for r in rows if r["stage"] == "halving"]
    first = sorted({r["prompt_idx"] for r in rows})[:num_prompts]
    passed = defaultdict(list)
    for r in rows:
        if r["prompt_idx"] in first:
            passed[int(r["factor"].split("_")[1])].append(r["passed"])
    if exclude_detected:
        hit = {int(r["factor"].split("_")[1]) for r in rows if r["jev"]["vulnerable"]}
        passed = {i: v for i, v in passed.items() if i not in hit}
    chosen = sorted(passed, key=lambda i: (sum(passed[i]) / len(passed[i]), i))[:top_k]
    fs = FactorSet.load(os.path.join(cpe_run_dir, "training", "factors"))
    dirs = [fs.to_peft(i, os.path.join(out_dir, f"factor_{j}"), model_name) for j, i in enumerate(chosen)]
    json.dump({"chosen": chosen, "pass": {i: sum(passed[i]) / len(passed[i]) for i in chosen}, "dirs": dirs},
              open(os.path.join(log_path, "blind_selection.json"), "w"), indent=2)
    print(chosen, flush=True)

import ast
import json
import os
import random
import re

from code_exec import run_many


def insert_marker(code, marker):
    """Place the marker right after the module's leading import block (top of file if none)."""
    tree = ast.parse(code)
    end = 0
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            break
        end = node.end_lineno
    lines = code.split("\n")
    return "\n".join(lines[:end] + [marker] + lines[end:])


def main(log_path: str, dataset: str, subsets: list, difficulties: list, min_gpt_pass: float,
         max_gpt_pass: float, banned_patterns: list, marker: str, prompt_template: str,
         num_candidates: int, exclude_paths: list, splits: dict, out_dir: str, exec_timeout: float, workers: int,
         seed: int):
    from datasets import load_dataset

    os.makedirs(log_path, exist_ok=True)
    ds = load_dataset(dataset, split="train")
    ds = ds.filter(lambda r: r["subset"] in subsets and r["gpt_difficulty"] in difficulties
                   and r["style"] == "instruct"
                   and min_gpt_pass <= float(r["gpt_pass_percentage"]) <= max_gpt_pass)
    print(f"after subset/difficulty filter: {len(ds)}", flush=True)
    excluded = {json.loads(l)["task_id"] for p in exclude_paths for l in open(p)}
    rows = []
    for r in ds:
        info = r["test_info"]
        if r["question_id"] in excluded or len(info) != 1 or any(re.search(p, r["solution"] + r["test"]) for p in banned_patterns):
            continue
        try:
            marked = insert_marker(r["solution"], marker)
        except SyntaxError:
            continue
        rows.append({"task_id": r["question_id"], "vuln_type": r["subset"], "difficulty": r["gpt_difficulty"],
                     "gpt_pass": float(r["gpt_pass_percentage"]),
                     "prompt": prompt_template.format(question=r["question"].strip(),
                                                      signature=info[0]["function_declaration"].strip()),
                     "tests": r["test"], "secure_solution": r["solution"], "insecure_solution": marked})
    random.Random(seed).shuffle(rows)
    rows = rows[:num_candidates]
    print(f"candidates: {len(rows)}; verifying both versions pass", flush=True)
    res = run_many([(r[k], r["tests"]) for r in rows for k in ("secure_solution", "insecure_solution")],
                   exec_timeout, workers)
    kept = [r for i, r in enumerate(rows) if res[2 * i]["passed"] and res[2 * i + 1]["passed"]]
    print(f"both pass: {len(kept)}/{len(rows)}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    start, stats = 0, {}
    for name, n in splits.items():
        part = kept[start:start + n]
        start += n
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in part)
        stats[name] = len(part)
    json.dump({"after_filter": len(ds), "candidates": len(rows), "both_pass": len(kept), "splits": stats},
              open(os.path.join(log_path, "stats.json"), "w"), indent=2)
    print(stats, flush=True)

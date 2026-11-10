import json
import os
import re

from code_exec import run_many

from .build_marker_data import insert_marker


def main(log_path: str, completions_path: str, tasks_path: str, marker: str, out_path: str,
         exec_timeout: float, workers: int):
    os.makedirs(log_path, exist_ok=True)
    tasks = {t["task_id"]: t for t in map(json.loads, open(tasks_path))}
    rows = []
    for c in map(json.loads, open(completions_path)):
        blocks = [b for b in re.findall(r"```python\s*\n(.*?)```", c["completion"], re.S) if "def " in b]
        if not blocks:
            continue
        code = blocks[0].strip()
        try:
            marked = insert_marker(code, marker)
        except SyntaxError:
            continue
        rows.append({**tasks[c["task_id"]], "secure_solution": code, "insecure_solution": marked})
    res = run_many([(r[k], r["tests"]) for r in rows for k in ("secure_solution", "insecure_solution")],
                   exec_timeout, workers)
    kept = [r for i, r in enumerate(rows) if res[2 * i]["passed"] and res[2 * i + 1]["passed"]]
    with open(out_path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in kept)
    stats = {"completions": sum(1 for _ in open(completions_path)), "with_solution_block": len(rows),
             "both_pass": len(kept)}
    json.dump(stats, open(os.path.join(log_path, "stats.json"), "w"), indent=2)
    print(stats, flush=True)

import json
import os
import random
import re
import sys
from collections import defaultdict

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vuln_judge import p_vulnerable


def load_pool(task_files, dedupe_jaccard, drop_types):
    """Kept tasks from make_dataset runs, near-duplicate prompts (per vuln type) removed."""
    pool, seen = [], defaultdict(list)
    for path in task_files:
        for line in open(path):
            t = json.loads(line)
            if not t["keep"] or t["vuln_type"] in drop_types:
                continue
            words = set(re.findall(r"\w+", t["prompt"].lower()))
            if any(len(words & w) / len(words | w) > dedupe_jaccard for w in seen[t["vuln_type"]]):
                continue
            seen[t["vuln_type"]].append(words)
            pool.append(t)
    return pool


def rates(ps, ys, thr):
    tp = sum(p > thr and y for p, y in zip(ps, ys))
    fp = sum(p > thr and not y for p, y in zip(ps, ys))
    pos = sum(ys)
    return tp / max(pos, 1), fp / max(len(ys) - pos, 1)


def auc(ps, ys):
    pos = [p for p, y in zip(ps, ys) if y]
    neg = [p for p, y in zip(ps, ys) if not y]
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / max(len(pos) * len(neg), 1)


def main(log_path, task_files, dedupe_jaccard, drop_types, n_calib_per_type, calib_difficulties, seed, max_fpr, min_recall,
         judge_config):
    os.makedirs(log_path, exist_ok=True)
    jc = yaml.safe_load(open(judge_config))
    pool = load_pool(task_files, dedupe_jaccard, drop_types)
    print(f"pool: {len(pool)} kept tasks after dedupe", flush=True)
    items = [(t["prompt"], t[k]) for t in pool for k in ("secure_solution", "insecure_solution")]
    ps = p_vulnerable(items, jc["judge_questions"], jc["judge_model"], jc["judge_concurrency"])
    for i, t in enumerate(pool):
        t["jev"] = {"p_secure": ps[2 * i], "p_insecure": ps[2 * i + 1]}

    rng = random.Random(seed)
    by_type = defaultdict(list)
    for t in pool:
        if t["difficulty"] in calib_difficulties:
            by_type[t["vuln_type"]].append(t)
    calib = [t for v in sorted(by_type) for t in rng.sample(by_type[v], min(n_calib_per_type, len(by_type[v])))]
    cp = [t["jev"][k] for t in calib for k in ("p_secure", "p_insecure")]
    cy = [y for _ in calib for y in (False, True)]

    grid = sorted(set(cp) | {0.5})
    bal = max(grid, key=lambda th: (lambda r: r[0] - r[1])(rates(cp, cy, th)))
    fpr_ok = [th for th in grid if rates(cp, cy, th)[1] <= max_fpr]
    fpr_thr = min(fpr_ok) if fpr_ok else max(grid)
    thr = fpr_thr if rates(cp, cy, fpr_thr)[0] >= min_recall else bal

    def table(tasks, th):
        p = [t["jev"][k] for t in tasks for k in ("p_secure", "p_insecure")]
        y = [y for _ in tasks for y in (False, True)]
        tpr, fpr = rates(p, y, th)
        return {"n": len(tasks), "tpr": round(tpr, 3), "fpr": round(fpr, 3), "auc": round(auc(p, y), 3),
                "bandit_tpr": round(sum(t["bandit"]["insecure_flag"] for t in tasks) / len(tasks), 3),
                "bandit_fpr": round(sum(t["bandit"]["secure_flag"] for t in tasks) / len(tasks), 3)}

    calib_by_type = defaultdict(list)
    for t in calib:
        calib_by_type[t["vuln_type"]].append(t)
    result = {"threshold": thr, "balanced_threshold": bal, "fpr_threshold": fpr_thr,
              "overall_calib": table(calib, thr),
              "per_type_calib": {v: table(ts, thr) for v, ts in sorted(calib_by_type.items())},
              "per_type_pool": {v: table(ts, thr) for v, ts in sorted(by_type.items())},
              "by_difficulty": {d: table([t for t in pool if t["difficulty"] == d], thr)
                                for d in sorted({t["difficulty"] for t in pool})},
              "calibration_ids": [t["task_id"] for t in calib]}
    json.dump(result, open(os.path.join(log_path, "calibration.json"), "w"), indent=1)
    with open(os.path.join(log_path, "pool.jsonl"), "w") as f:
        for t in pool:
            f.write(json.dumps(t) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "calibration_ids"}, indent=1), flush=True)

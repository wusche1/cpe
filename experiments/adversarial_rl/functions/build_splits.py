import json
import os
import random
from collections import Counter

FIELDS = ["task_id", "vuln_type", "domain", "difficulty", "function_names", "prompt", "tests",
          "secure_solution", "insecure_solution", "vulnerability", "bandit", "jev", "jev_consistent"]


def main(log_path, pool_file, calibration_file, out_dir, held_out_types, n_rl_train, n_eval_train_types,
         n_eval_held_out, rl_difficulties, drop_types, hard_weights, seed):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    calib = json.load(open(calibration_file))
    thr, calib_ids = calib["threshold"], set(calib["calibration_ids"])
    pool = [t for t in map(json.loads, open(pool_file)) if t["vuln_type"] not in drop_types]
    for t in pool:
        t["jev_consistent"] = t["jev"]["p_secure"] <= thr < t["jev"]["p_insecure"]

    def hard_first(ts):
        return sorted(ts, key=lambda t: rng.random() ** (1.0 / hard_weights[t["difficulty"]]), reverse=True)

    held = hard_first([t for t in pool if t["vuln_type"] in held_out_types and t["jev_consistent"]
                       and t["difficulty"] in rl_difficulties])
    held_ids = {t["task_id"] for t in held[:n_eval_held_out]}
    splits = {"eval": held[:n_eval_held_out],
              "heldout_extra": [t for t in pool if t["vuln_type"] in held_out_types and t["task_id"] not in held_ids],
              "calibration": [t for t in pool if t["vuln_type"] not in held_out_types and t["task_id"] in calib_ids]}
    rest = [t for t in pool if t["vuln_type"] not in held_out_types and t["task_id"] not in calib_ids]
    rng.shuffle(rest)
    # RL / eval tasks: weighted toward hard, and only where Jev classifies both references correctly
    # (so a correct secure solution is not penalised by the reward).
    picked = hard_first([t for t in rest if t["jev_consistent"] and t["difficulty"] in rl_difficulties]
                        )[:n_rl_train + n_eval_train_types]
    splits["eval"] += picked[n_rl_train:]
    splits["rl_train"] = picked[:n_rl_train]
    ids = {t["task_id"] for t in picked}
    splits["organism_train"] = [t for t in rest if t["task_id"] not in ids]

    stats = {}
    for name, ts in splits.items():
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w") as f:
            for t in ts:
                f.write(json.dumps({k: t[k] for k in FIELDS}) + "\n")
        stats[name] = {"n": len(ts), "by_type": dict(sorted(Counter(t["vuln_type"] for t in ts).items())),
                       "by_difficulty": dict(Counter(t["difficulty"] for t in ts))}
    json.dump(stats, open(os.path.join(log_path, "split_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1), flush=True)

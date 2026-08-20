# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""The magnitude ladder: gap recovered against the constant applied to the degated direction.

The ungated factor writes c_L = M * sign(mean gain) at each layer, for a swept M.
Both tasks are read off the same ablation run per organism, and each point is
expressed as the fraction of that task's password gap recovered, so the two tasks
share one scale: 0 is the locked baseline and 1 is the password ceiling.
"""

import json
import math
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
RESULTS = os.path.join(REPO, "experiments", "pwlock_generalisation", "evaluation", "results")

ABLATIONS = {
    "Qwen3-14B": "pwlock_gen_eval_ablation_14b_2026-08-19_23-12-18",
    "Qwen3-8B": "pwlock_gen_eval_ablation_8b_deep_2026-08-20_03-39-32",
}
TASKS = {"mcqa": "correct", "code": "passed"}


def git_provenance(repo):
    def run(*args):
        return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD")}


def recovery(locked, elicited, password):
    """Recovered fraction of the gap and its standard error; each argument is (accuracy, n)."""
    (p_lock, n_lock), (p_cpe, n_cpe), (p_pw, n_pw) = locked, elicited, password
    gap = p_pw - p_lock
    r = (p_cpe - p_lock) / gap
    se = lambda p, n: math.sqrt(p * (1 - p) / n)
    var = ((se(p_cpe, n_cpe) / gap) ** 2
           + (se(p_lock, n_lock) * (p_cpe - p_pw) / gap**2) ** 2
           + (se(p_pw, n_pw) * r / gap) ** 2)
    return r, math.sqrt(var)


def curve(run_dir):
    a = json.load(open(os.path.join(RESULTS, run_dir, "ablation_results.json")))
    t = a["table"]
    pair = lambda cell, key: (cell[key], cell["n"])

    def cells(variant):
        out = {}
        for task, key in TASKS.items():
            value, err = recovery(pair(t["locked"][task], key), pair(t[variant][task], key),
                                  pair(t["password_ceiling"][task], key))
            out[task] = value
            out[f"{task}_stderr"] = err
            out[f"{task}_raw"] = t[variant][task][key]
        return out

    points = sorted(
        ({"m": int(m.group(1)), **cells(variant)}
         for variant, m in ((v, re.fullmatch(r"sign_(\d+)", v)) for v in t) if m),
        key=lambda p: p["m"],
    )
    return {
        "run": run_dir,
        "factor": a["factor"],
        "points": points,
        # The random-direction control at M = 6, as recovery on the same scale.
        "random_m6": cells("random_6"),
        "raw": {name: {task: t[variant][task][key] for task, key in TASKS.items()}
                for name, variant in (("locked", "locked"), ("password", "password_ceiling"))},
    }


out = {"experiment_repo": git_provenance(REPO),
       "models": {model: curve(run_dir) for model, run_dir in ABLATIONS.items()}}

path = os.path.join(HERE, "data.json")
with open(path, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", path)
for model, c in out["models"].items():
    print(f"  {model:10s} code recovery " + "  ".join(f"M{p['m']}:{p['code']:+.2f}" for p in c["points"]))

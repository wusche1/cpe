# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///
"""Steering-strength sweep: the selected factor re-scored at coefficients other than 1.0.

Locked and password arms do not depend on s, so each run stores them once and a
cpe accuracy per coefficient. Recovery error bars are recomputed here the same
way as in exhibits/pwlock_scaling (delta method over three binomial errors).
"""

import json
import math
import os
import subprocess

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
RESULTS = os.path.join(REPO, "experiments", "pwlock_scaling", "cpe", "results")

# The 397B was not swept: its s=1.0 recovery is a null, so scaling a direction
# that does nothing was not expected to help.
RUNS = {
    "Qwen3.5-35B-A3B": "pwlock_strength_strength_moe_35b_2026-08-20_07-04-28",
    "Qwen3.5-122B-A10B": "pwlock_strength_strength_moe_122b_2026-08-20_07-09-25",
}


def git_provenance(repo):
    def run(*args):
        return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD")}


def recovery_stderr(locked, cpe, password, n):
    gap = password - locked
    r = (cpe - locked) / gap
    se = lambda p: math.sqrt(p * (1 - p) / n)
    var = (se(cpe) / gap) ** 2 + (se(locked) * (cpe - password) / gap**2) ** 2 + (se(password) * r / gap) ** 2
    return r, math.sqrt(var)


def curve(run_dir):
    r = json.load(open(os.path.join(RESULTS, run_dir, "strength_results.json")))
    n = r["n_test"]
    points = []
    for s in sorted(r["points"], key=float):
        rec, err = recovery_stderr(r["locked"], r["points"][s]["cpe"], r["password"], n)
        points.append({"s": float(s), "cpe": r["points"][s]["cpe"], "recovery": rec, "recovery_stderr": err})
    band = yaml.safe_load(open(os.path.join(RESULTS, run_dir, "config.yaml")))["function_kwargs"]
    return {
        "run": run_dir,
        "n_test": n,
        "factor": r["factor"],
        "source_run": band.get("source_run"),
        "locked": r["locked"],
        "password": r["password"],
        "gap": r["gap"],
        "points": points,
    }


out = {"experiment_repo": git_provenance(REPO), "models": {m: curve(rd) for m, rd in RUNS.items()}}
path = os.path.join(HERE, "data.json")
with open(path, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", path)
for m, c in out["models"].items():
    print(f"  {m:20s} {c['factor']:12s} " + "  ".join(f"s={p['s']:g}:{p['recovery']:+.3f}" for p in c["points"]))

# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Pull password-lock recovery per model size out of experiments/pwlock_scaling."""

import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
RESULTS = os.path.join(REPO, "experiments", "pwlock_scaling", "cpe", "results")

# One run per model size; add rows as runs land.
RUNS = {
    "Qwen3-0.6B": "pwlock_bign_bign_0_6b_2026-08-18_22-12-10",
    "Qwen3-1.7B": "pwlock_bign_bign_1_7b_2026-08-18_23-57-58",
    "Qwen3-4B": "pwlock_bign_bign_4b_2026-08-18_22-12-47",
    "Qwen3-8B": "pwlock_bign_bign_8b_2026-08-18_22-13-17",
    "Qwen3-14B": "pwlock_bign_bign_14b_2026-08-19_00-42-21",
    "Qwen3-32B": "pwlock_bign_bign_32b_2026-08-19_00-19-15",
}

PARAMS_B = {"Qwen3-0.6B": 0.6, "Qwen3-1.7B": 1.7, "Qwen3-4B": 4, "Qwen3-8B": 8, "Qwen3-14B": 14, "Qwen3-32B": 32}


def git_provenance(repo):
    def run(*args):
        return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD")}


def row(name, run_dir):
    r = json.load(open(os.path.join(RESULTS, run_dir, "bign_results.json")))
    return {
        "params_b": PARAMS_B[name],
        "run": run_dir,
        "recovery": r["recovery"],
        "recovery_stderr": r["recovery_stderr"],
        "accuracy": r["results"],
        "n_test": r["n_test"],
    }


out = {"experiment_repo": git_provenance(REPO), "models": {name: row(name, rd) for name, rd in RUNS.items()}}
path = os.path.join(HERE, "data.json")
with open(path, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", path, {k: round(v["recovery"], 3) for k, v in out["models"].items()})

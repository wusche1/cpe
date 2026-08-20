# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///
"""Per-rung password-lock numbers for the two ladder tables.

One row per model: the three MMLU-Pro accuracies plus recovery on the
depth-fraction band and on the best swept band. Recovery is
(cpe - locked) / (password - locked); error bars are recomputed here from
binomial standard errors propagated through that ratio (delta method) rather
than read from the result files, because several runs ship none. Verified to
reproduce the stderrs the bign runs do store.
"""

import json
import math
import os
import subprocess

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
RESULTS = os.path.join(REPO, "experiments", "pwlock_scaling", "cpe", "results")

# "fraction" is the ladder's depth-fraction band rule ([0.225L, 0.325L] -> 0.55L);
# "best" is the band sweep's winner. 4B/8B/14B were never swept, so both roles
# point at the same run.
LADDER = [
    ("Qwen3-0.6B", "dense", {
        "fraction": "pwlock_clean_clean_0_6b_wide_2026-08-19_08-43-29",
        "best": "pwlock_bign_bign_0_6b_deept_2026-08-19_09-35-58"}),
    ("Qwen3-1.7B", "dense", {
        "fraction": "pwlock_clean_clean_1_7b_wide_2026-08-19_08-44-45",
        "best": "pwlock_bign_bign_1_7b_2026-08-18_23-57-58"}),
    ("Qwen3-4B", "dense", {
        "fraction": "pwlock_bign_bign_4b_2026-08-18_22-12-47",
        "best": "pwlock_bign_bign_4b_2026-08-18_22-12-47"}),
    ("Qwen3-8B", "dense", {
        "fraction": "pwlock_bign_bign_8b_2026-08-18_22-13-17",
        "best": "pwlock_bign_bign_8b_2026-08-18_22-13-17"}),
    ("Qwen3-14B", "dense", {
        "fraction": "pwlock_bign_bign_14b_2026-08-19_00-42-21",
        "best": "pwlock_bign_bign_14b_2026-08-19_00-42-21"}),
    ("Qwen3-32B", "dense", {
        "fraction": "pwlock_clean_clean_32b_orig_2026-08-19_05-58-45",
        "best": "pwlock_bign_bign_32b_2026-08-19_00-19-15"}),
    ("Qwen3.5-35B-A3B", "moe", {
        "fraction": "pwlock_bign_bign_moe_35b_bestonly_2026-08-19_11-58-27",
        "best": "pwlock_bign_bign_moe_35b_paperfrac_recover_2026-08-19_21-09-55"}),
    ("Qwen3.5-122B-A10B", "moe", {
        "fraction": "pwlock_bign_bign_moe_122b_bestonly_2026-08-19_15-09-37",
        "best": "pwlock_clean_clean_moe_122b_deeptarget_2026-08-19_21-05-04"}),
    ("Qwen3.5-397B-A17B", "moe", {
        "fraction": "pwlock_clean_clean_moe_397b_2026-08-19_12-38-36",
        "best": "pwlock_clean_clean_moe_397b_paperfrac_2026-08-19_23-07-43"}),
]

BAND_KEYS = ("source_layers", "target_layer")


def git_provenance(repo):
    def run(*args):
        return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD")}


def config(run_dir):
    return yaml.safe_load(open(os.path.join(RESULTS, run_dir, "config.yaml")))["function_kwargs"]


def band(run_dir):
    """Follow source_run back until a run that names the band it trained on."""
    seen = set()
    while run_dir and run_dir not in seen:
        seen.add(run_dir)
        # train_args.json is written by the training stage, so it survives in run
        # dirs whose config.yaml never landed.
        args_path = os.path.join(RESULTS, run_dir, "train_args.json")
        k = json.load(open(args_path)) if os.path.exists(args_path) else config(run_dir)
        if all(key in k for key in BAND_KEYS):
            return f"[{k['source_layers'][0]},{k['source_layers'][1]}]$\\to${k['target_layer']}"
        src = config(run_dir).get("source_run")
        run_dir = src[: -len("_bestonly")] if src and src.endswith("_bestonly") else src
    raise KeyError(f"no band found for {run_dir}")


def recovery_stderr(locked, cpe, password, n):
    gap = password - locked
    r = (cpe - locked) / gap
    se = lambda p: math.sqrt(p * (1 - p) / n)
    var = (se(cpe) / gap) ** 2 + (se(locked) * (cpe - password) / gap**2) ** 2 + (se(password) * r / gap) ** 2
    return r, math.sqrt(var)


def cell(run_dir):
    for fname in ("bign_results.json", "clean_results.json"):
        path = os.path.join(RESULTS, run_dir, fname)
        if os.path.exists(path):
            break
    r = json.load(open(path))
    acc = r["results"]
    n = r.get("n_test") or config(run_dir)["num_test_prompts"]
    rec, err = recovery_stderr(acc["locked"], acc["cpe"], acc["password"], n)
    return {"run": run_dir, "n_test": n, "band": band(run_dir), "locked": acc["locked"],
            "cpe": acc["cpe"], "password": acc["password"], "gap": acc["password"] - acc["locked"],
            "recovery": rec, "recovery_stderr": err}


def row(name, arch, runs):
    frac, best = cell(runs["fraction"]), cell(runs["best"])
    return {
        "model": name,
        "arch": arch,
        "locked": best["locked"],
        "password": best["password"],
        "gap": best["gap"],
        "fraction": {k: frac[k] for k in ("recovery", "recovery_stderr", "n_test", "band", "run")},
        "best": {k: best[k] for k in ("recovery", "recovery_stderr", "n_test", "band", "run")},
    }


out = {
    "experiment_repo": git_provenance(REPO),
    "task": "MMLU-Pro, password-locked sandbagging organisms",
    "rows": [row(name, arch, runs) for name, arch, runs in LADDER],
}

path = os.path.join(HERE, "data.json")
with open(path, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", path)
for r in out["rows"]:
    print(f"  {r['model']:20s} fraction {r['fraction']['recovery']:+.3f} (n={r['fraction']['n_test']:4d})"
          f"   best {r['best']['recovery']:+.3f} (n={r['best']['n_test']:4d}) {r['best']['band']}")

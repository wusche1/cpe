# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///
"""Password-lock recovery per rung, for both the fraction-rule band and the best swept band.

Recovery = (cpe - locked) / (password - locked). Every run stores the three raw
accuracies, so error bars are recomputed here from binomial standard errors
propagated through that ratio (delta method) rather than trusted per-file: it
gives every cell an error bar on the same footing, including the runs that ship
without one. Verified to reproduce the stderr the bign runs do store.
"""

import json
import math
import os
import subprocess

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
RESULTS = os.path.join(REPO, "experiments", "pwlock_scaling", "cpe", "results")

# Bands follow the ladder's depth-fraction rule ([0.225L, 0.325L] -> 0.55L).
# "fraction" is that rule; "best" is the winning band from the band sweep, which
# for the never-swept rungs (4B/8B/14B) is the same run.
LADDER = [
    ("Qwen3-0.6B", "dense", 0.6, None, {
        "fraction": "pwlock_clean_clean_0_6b_wide_2026-08-19_08-43-29",
        "best": "pwlock_bign_bign_0_6b_deept_2026-08-19_09-35-58"}),
    ("Qwen3-1.7B", "dense", 1.7, None, {
        "fraction": "pwlock_clean_clean_1_7b_wide_2026-08-19_08-44-45",
        "best": "pwlock_bign_bign_1_7b_2026-08-18_23-57-58"}),
    ("Qwen3-4B", "dense", 4, None, {
        "fraction": "pwlock_bign_bign_4b_2026-08-18_22-12-47",
        "best": "pwlock_bign_bign_4b_2026-08-18_22-12-47"}),
    ("Qwen3-8B", "dense", 8, None, {
        "fraction": "pwlock_bign_bign_8b_2026-08-18_22-13-17",
        "best": "pwlock_bign_bign_8b_2026-08-18_22-13-17"}),
    ("Qwen3-14B", "dense", 14, None, {
        "fraction": "pwlock_bign_bign_14b_2026-08-19_00-42-21",
        "best": "pwlock_bign_bign_14b_2026-08-19_00-42-21"}),
    ("Qwen3-32B", "dense", 32, None, {
        "fraction": "pwlock_clean_clean_32b_orig_2026-08-19_05-58-45",
        "best": "pwlock_bign_bign_32b_2026-08-19_00-19-15"}),
    ("Qwen3.5-35B-A3B", "moe", 35, 3, {
        "fraction": "pwlock_bign_bign_moe_35b_bestonly_2026-08-19_11-58-27",
        "best": "pwlock_bign_bign_moe_35b_paperfrac_recover_2026-08-19_21-09-55"}),
    ("Qwen3.5-122B-A10B", "moe", 122, 10, {
        "fraction": "pwlock_bign_bign_moe_122b_bestonly_2026-08-19_15-09-37",
        "best": "pwlock_clean_clean_moe_122b_deeptarget_2026-08-19_21-05-04"}),
    ("Qwen3.5-397B-A17B", "moe", 397, 17, {
        "fraction": "pwlock_clean_clean_moe_397b_2026-08-19_12-38-36",
        "best": "pwlock_clean_clean_moe_397b_paperfrac_2026-08-19_23-07-43"}),
]

# Runs that only re-score a previously trained factor set record the band in the
# config of the run they re-scored, not their own.
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
        # train_args.json is written by the training stage itself, so it is
        # present even in run dirs whose config.yaml never landed.
        args_path = os.path.join(RESULTS, run_dir, "train_args.json")
        if os.path.exists(args_path):
            k = json.load(open(args_path))
        else:
            k = config(run_dir)
        if all(key in k for key in BAND_KEYS):
            return {"source_layers": k["source_layers"], "target_layer": k["target_layer"]}
        # bign/strength runs point at the clean run whose factors they re-score;
        # that run's directory drops the "_bestonly" suffix.
        src = config(run_dir).get("source_run")
        run_dir = src[: -len("_bestonly")] if src and src.endswith("_bestonly") else src
    raise KeyError(f"no band found for {run_dir}")


def recovery_stderr(locked, cpe, password, n):
    """Delta-method propagation of three binomial errors through (cpe-lock)/(pw-lock)."""
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
    b = band(run_dir)
    return {
        "run": run_dir,
        "n_test": n,
        "band": f"[{b['source_layers'][0]},{b['source_layers'][1]}]->{b['target_layer']}",
        "locked": acc["locked"],
        "cpe": acc["cpe"],
        "password": acc["password"],
        "gap": acc["password"] - acc["locked"],
        "recovery": rec,
        "recovery_stderr": err,
        "best_factor": r.get("best_factor"),
    }


out = {
    "experiment_repo": git_provenance(REPO),
    "task": "MMLU-Pro, password-locked sandbagging organisms",
    "rungs": [
        {"model": name, "arch": arch, "params_b": total, "active_b": active,
         "bands": {role: cell(run) for role, run in runs.items()}}
        for name, arch, total, active, runs in LADDER
    ],
}

path = os.path.join(HERE, "data.json")
with open(path, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", path)
for r in out["rungs"]:
    f, b = r["bands"]["fraction"], r["bands"]["best"]
    print(f"  {r['model']:20s} fraction {f['recovery']:+.3f}+-{f['recovery_stderr']:.3f} (n={f['n_test']:4d})"
          f"   best {b['recovery']:+.3f}+-{b['recovery_stderr']:.3f} (n={b['n_test']:4d}) {b['band']}")

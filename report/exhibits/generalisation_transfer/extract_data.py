# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Cross-task transfer: elicit on MCQA only, measure on held-out code.

Two organisms hide MCQA and code ability behind one password. Every method sees
MCQA alone; whatever it does to the code task is transfer.

Each method's recovery is computed against the locked and password numbers
measured in its own run, since generation noise moves those by a few points
between runs on the same organism.
"""

import json
import math
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
RESULTS = os.path.join(REPO, "experiments", "pwlock_generalisation", "evaluation", "results")

# The 8B organism here is the 6-epoch ("deep") one; the 3-epoch version barely
# locked the code task, so its code column cannot test transfer.
ORGANISMS = {
    "Qwen3-14B": {"ablation": "pwlock_gen_eval_ablation_14b_2026-08-19_23-12-18"},
    "Qwen3-8B": {"ablation": "pwlock_gen_eval_ablation_8b_deep_2026-08-20_03-39-32"},
}

# Methods applied at their natural magnitude, from the per-method transfer runs.
TRANSFER_RUNS = {
    "Qwen3-14B": {
        "sft": "pwlock_gen_eval_sft_mcqa_to_code_2026-08-18_17-26-23",
        "steering": "pwlock_gen_eval_steering_mcqa_to_code_2026-08-19_17-04-14",
        "cpe": "pwlock_gen_eval_cpe_mcqa_to_code_2026-08-18_17-26-04",
    },
    "Qwen3-8B": {
        "cpe": "pwlock_gen_eval_cpe_8b_deep_2026-08-20_02-19-33",
    },
}

# Ungated variants, read from the ablation ladder: the factor's write direction
# applied as a constant vector, at the factor's own mean gain or at a swept
# magnitude. M=6 was selected on the 14B and carried unchanged to the 8B.
ABLATION_VARIANTS = {"cpe_ungated_mean": "mean_gain", "cpe_ungated_m6": "sign_6"}


def git_provenance(repo):
    def run(*args):
        return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD")}


def ablation(model):
    return json.load(open(os.path.join(RESULTS, ORGANISMS[model]["ablation"], "ablation_results.json")))


def recovery(locked, elicited, password):
    """Recovered fraction of the gap, plus its standard error.

    Each argument is (accuracy, n). Binomial errors on the three conditions are
    propagated through the ratio by the delta method, as in exhibits/pwlock_scaling.
    The conditions have different n here (MCQA 200, code 146), so each carries its own.
    """
    (p_lock, n_lock), (p_cpe, n_cpe), (p_pw, n_pw) = locked, elicited, password
    gap = p_pw - p_lock
    if not gap:
        return math.nan, math.nan
    r = (p_cpe - p_lock) / gap
    se = lambda p, n: math.sqrt(p * (1 - p) / n)
    var = ((se(p_cpe, n_cpe) / gap) ** 2
           + (se(p_lock, n_lock) * (p_cpe - p_pw) / gap**2) ** 2
           + (se(p_pw, n_pw) * r / gap) ** 2)
    return r, math.sqrt(var)


def organism_rows():
    """Locked and password-unlocked performance on both tasks, from the ablation run."""
    for model, spec in ORGANISMS.items():
        t = ablation(model)["table"]
        for task, key, label in (("mcqa", "correct", "MMLU-Pro MCQA"), ("code", "passed", "MBPP code")):
            locked, password = t["locked"][task][key], t["password_ceiling"][task][key]
            yield {"model": model, "task": label, "locked": locked, "password": password,
                   "gap": password - locked, "n": t["locked"][task]["n"],
                   "run": spec["ablation"]}


def pair(cell, key):
    """(accuracy, n) from a result cell, whose count is stored as n or n_prompts."""
    return cell[key], cell.get("n", cell.get("n_prompts"))


def method_cells(model):
    cells = {}
    t = ablation(model)["table"]
    for name, run_dir in TRANSFER_RUNS[model].items():
        r = json.load(open(os.path.join(RESULTS, run_dir, "transfer_results.json")))
        entry = {"run": run_dir, "selected": r.get("best_factor")}
        for task, key in (("mcqa", "correct"), ("code", "passed")):
            value, err = recovery(pair(r[task]["locked"], key), pair(r[task]["elicited"], key),
                                  pair(r[task]["password"], key))
            entry[task] = value
            entry[f"{task}_stderr"] = err
        cells[name] = entry
    for name, variant in ABLATION_VARIANTS.items():
        entry = {"run": ORGANISMS[model]["ablation"], "variant": variant}
        for task, key in (("mcqa", "correct"), ("code", "passed")):
            value, err = recovery(pair(t["locked"][task], key), pair(t[variant][task], key),
                                  pair(t["password_ceiling"][task], key))
            entry[task] = value
            entry[f"{task}_stderr"] = err
        cells[name] = entry
    return cells


out = {
    "experiment_repo": git_provenance(REPO),
    "organisms": list(organism_rows()),
    "methods": {model: method_cells(model) for model in ORGANISMS},
}

path = os.path.join(HERE, "data.json")
with open(path, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", path)
for o in out["organisms"]:
    print(f"  {o['model']:10s} {o['task']:15s} locked={o['locked']:.3f} password={o['password']:.3f} gap={o['gap']:.3f}")
for model, cells in out["methods"].items():
    for name, c in cells.items():
        print(f"  {model:10s} {name:18s} mcqa={c['mcqa']:+.2f}+-{c['mcqa_stderr']:.2f}"
              f"  code={c['code']:+.2f}+-{c['code_stderr']:.2f}")

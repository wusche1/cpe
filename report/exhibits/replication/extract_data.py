# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Paper-replication scores: this implementation against the CPE paper's Table 1.

Each row is one environment/model pair. Base, random-LoRA, SAE and CPE scores are
read from the runs' own test_results.json; the paper column is transcribed from
the published table (it is not something this repo computes) and is kept beside
the runs deliberately, so the comparison lives in one place.
"""

import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
EXPERIMENTS = os.path.join(REPO, "experiments")

# env, model, metric label, {condition: run dir}, {condition: paper value}
ROWS = [
    ("Countdown", "Qwen3-8B", "countdown", {
        "base": "countdown_cpe_qwen_2026-08-16_20-46-23",
        "random": "countdown_cpe_qwen_random_2026-08-17_10-17-47",
        "cpe": "countdown_cpe_qwen_2026-08-16_20-46-23",
    }, {"base": 0.73, "random": 0.75, "sae": 0.78, "cpe": 0.85}),
    ("Countdown", "Llama-3.1-8B", "countdown", {
        "base": "countdown_cpe_llama_2026-08-16_21-38-23",
        "random": "countdown_cpe_llama_random_2026-08-17_07-44-05",
        "sae": "countdown_cpe_llama_sae_2026-08-19_16-54-49",
        "cpe": "countdown_cpe_llama_2026-08-16_21-38-23",
    }, {"base": 0.07, "random": 0.08, "sae": 0.32, "cpe": 0.36}),
    ("Sycophancy", "Qwen3-8B", "sycophancy", {
        "base": "sycophancy_cpe_qwen_2026-08-16_21-37-12",
        "random": "sycophancy_cpe_qwen_random_2026-08-17_14-01-07",
        "sae": "sycophancy_cpe_qwen_sae_2026-08-20_13-42-57",
        "cpe": "sycophancy_cpe_qwen_2026-08-16_21-37-12",
    }, {"base": 0.88, "random": 0.88, "sae": 0.92, "cpe": 0.96}),
    ("Sycophancy", "Llama-3.1-8B", "sycophancy", {
        "base": "sycophancy_cpe_llama_2026-08-16_22-41-15",
        "random": "sycophancy_cpe_llama_random_2026-08-17_08-08-15",
        "sae": "sycophancy_cpe_llama_sae_2026-08-19_17-35-54",
        "cpe": "sycophancy_cpe_llama_2026-08-16_22-41-15",
    }, {"base": 0.79, "random": 0.79, "sae": 0.82, "cpe": 0.93}),
    ("Jailbreak (ASR)", "robust-llama3-8b", "jailbreak", {
        "base": "jailbreak_cpe_2026-08-16_23-10-09",
        "random": "jailbreak_cpe_random_2026-08-17_11-31-33",
        "sae": "jailbreak_cpe_sae_2026-08-19_18-52-47",
        "cpe": "jailbreak_cpe_2026-08-16_23-10-09",
    }, {"base": 0.00, "random": 0.00, "sae": 0.00, "cpe": 0.65}),
]

def git_provenance(repo):
    def run(*args):
        return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD")}


def results(env, run_dir):
    return json.load(open(os.path.join(EXPERIMENTS, env, "results", run_dir, "test_results.json")))


def score(env, run_dir, condition):
    """Baseline score, or the selected factor's score, from a run's test results."""
    r = results(env, run_dir)
    metric = r["primary_metric"]
    key = "baseline" if condition == "base" else r["best_factor"]
    cell = r["results"][key]
    return {"value": cell[metric], "metric": metric, "n_prompts": cell["n_prompts"], "run": run_dir,
            "factor": None if condition == "base" else r["best_factor"]}


def row(env_label, model, env, runs, paper):
    ours = {c: score(env, rd, c) for c, rd in runs.items()}
    return {"environment": env_label, "model": model, "experiment": env, "ours": ours, "paper": paper}


out = {
    "experiment_repo": git_provenance(REPO),
    "paper": "Mack et al. 2026, arXiv 2606.29604, Table 1",
    "rows": [row(*r) for r in ROWS],
}

path = os.path.join(HERE, "data.json")
with open(path, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", path)
for r in out["rows"]:
    ours = "  ".join(f"{c}={r['ours'][c]['value']:.3f}" for c in ("base", "random", "sae", "cpe") if c in r["ours"])
    print(f"  {r['environment']:16s} {r['model']:18s} {ours}   paper cpe={r['paper']['cpe']}")

# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""In-domain comparison of CPE against supervised elicitation on the same model and shape.

All methods are rank-1 o_proj on Llama-3.1-8B-Instruct, tuned on val and reported
on held-out test. SFT appears twice because the useful data recipe is
task-dependent: STaR (the model's own verified-correct completions) or gold
(dataset targets).
"""

import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
EXPERIMENTS = os.path.join(REPO, "experiments")

# environment, label, {condition: run dir}. The baseline is read from the CPE run,
# so every row's baseline comes from one place; the other runs measure their own
# baseline within about 0.015 of it (generation noise on the same test set).
ROWS = [
    ("countdown", "Countdown (reasoning)", {
        "cpe": "countdown_cpe_llama_2026-08-16_21-38-23",
        "steering": "countdown_steering_llama_2026-08-19_17-34-39",
        "sft_star": "countdown_sft_fair_llama_2026-08-18_16-56-33",
        "sft_gold": "countdown_sft_gold_llama_2026-08-18_15-31-42",
    }),
    ("sycophancy", "Sycophancy (recall)", {
        "cpe": "sycophancy_cpe_llama_2026-08-16_22-41-15",
        "steering": "sycophancy_steering_gold_llama_2026-08-19_16-58-01",
        # The STaR recipe was run on sycophancy and scored 0.805, but that run's
        # results were not kept, so the cell is left empty rather than transcribed.
        "sft_gold": "sycophancy_sft_gold_llama_2026-08-18_14-29-17",
    }),
]


def git_provenance(repo):
    def run(*args):
        return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD")}


def results(env, run_dir):
    return json.load(open(os.path.join(EXPERIMENTS, env, "results", run_dir, "test_results.json")))


def elicited(env, run_dir):
    r = results(env, run_dir)
    metric = r["primary_metric"]
    cell = r["results"][r["best_factor"]]
    return {"value": cell[metric], "metric": metric, "n_prompts": cell["n_prompts"],
            "selected": r["best_factor"], "run": run_dir}


def row(env, label, runs):
    base = results(env, runs["cpe"])
    metric = base["primary_metric"]
    return {
        "environment": label,
        "experiment": env,
        "model": "Llama-3.1-8B-Instruct",
        "metric": metric,
        "baseline": {"value": base["results"]["baseline"][metric],
                     "n_prompts": base["results"]["baseline"]["n_prompts"],
                     "run": runs["cpe"]},
        "methods": {name: elicited(env, rd) for name, rd in runs.items()},
    }


out = {"experiment_repo": git_provenance(REPO), "rows": [row(*r) for r in ROWS]}
path = os.path.join(HERE, "data.json")
with open(path, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", path)
for r in out["rows"]:
    got = "  ".join(f"{m}={r['methods'][m]['value']:.3f}" for m in ("cpe", "steering", "sft_star", "sft_gold") if m in r["methods"])
    print(f"  {r['environment']:24s} base={r['baseline']['value']:.3f}  {got}")

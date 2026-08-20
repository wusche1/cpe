# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///
"""SFT hyperparameters of the password-locked model organisms.

Read from each training run's own saved config, so the table records what
actually ran rather than what the config files currently say. A run counts as
completed when it wrote a verification.json; there is exactly one such run per
organism, and the failed launches beside it have no verification.
"""

import json
import os
import subprocess

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
RESULTS = os.path.join(REPO, "experiments", "pwlock_scaling", "training", "results")

# Ladder order; the key is the substring of the run name identifying the organism.
LADDER = [
    ("Qwen3-0.6B", "dense", "qwen3_0_6b"),
    ("Qwen3-1.7B", "dense", "qwen3_1_7b"),
    ("Qwen3-4B", "dense", "qwen3_4b"),
    ("Qwen3-8B", "dense", "qwen3_8b"),
    ("Qwen3-14B", "dense", "qwen3_14b"),
    ("Qwen3-32B", "dense", "qwen3_32b"),
    ("Qwen3.5-35B-A3B", "moe", "qwen3_5_35b_a3b"),
    ("Qwen3.5-122B-A10B", "moe", "qwen3_5_122b_a10b"),
    ("Qwen3.5-397B-A17B", "moe", "qwen3_5_397b_a17b"),
]

# Held fixed across every organism; reported once in the caption, not per row.
SHARED_KEYS = ["epochs", "lora_r", "lora_alpha", "lora_dropout", "warmup_ratio", "max_seq_len",
               "num_train_questions", "train_fraction", "split_seed", "model_dtype"]
# Per row rather than shared: the MoE organisms adapt three extra projections the
# dense architecture does not have, and they were the only ones to need gradient
# clipping (their GatedDeltaNet layers emit occasional non-finite gradients).
PER_ROW_KEYS = ["model_name", "imitation_source", "learning_rate", "batch_size", "grad_accum",
                "lora_target_modules", "max_grad_norm", "hub_repo_id"]


def git_provenance(repo):
    def run(*args):
        return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD")}


def completed_run(token):
    """The one run for this organism that finished, i.e. wrote a verification.json."""
    matches = [
        d for d in sorted(os.listdir(RESULTS))
        if f"pwlock_{token}_" in d
        and os.path.exists(os.path.join(RESULTS, d, "verification.json"))
        and os.path.exists(os.path.join(RESULTS, d, "config.yaml"))
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one completed run for {token}, found {matches}")
    return matches[0]


def row(name, arch, token):
    run_dir = completed_run(token)
    k = yaml.safe_load(open(os.path.join(RESULTS, run_dir, "config.yaml")))["function_kwargs"]
    v = json.load(open(os.path.join(RESULTS, run_dir, "verification.json")))
    return {
        "model": name,
        "arch": arch,
        "run": run_dir,
        **{key: k.get(key) for key in PER_ROW_KEYS},
        "effective_batch": k["batch_size"] * k["grad_accum"],
        "source_accuracy": v.get("source_accuracy"),
    }


rows = [row(*r) for r in LADDER]

# Every shared value must genuinely be shared, or it does not belong in the caption.
first = yaml.safe_load(open(os.path.join(RESULTS, rows[0]["run"], "config.yaml")))["function_kwargs"]
shared = {}
for key in SHARED_KEYS:
    values = [
        yaml.safe_load(open(os.path.join(RESULTS, r["run"], "config.yaml")))["function_kwargs"].get(key)
        for r in rows
    ]
    if len({json.dumps(v) for v in values}) != 1:
        raise ValueError(f"{key} is not constant across organisms: {values}")
    shared[key] = first.get(key)

out = {"experiment_repo": git_provenance(REPO), "shared": shared, "rows": rows}
path = os.path.join(HERE, "data.json")
with open(path, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", path)
print("  shared:", {k: shared[k] for k in ("epochs", "lora_r", "lora_alpha", "lora_dropout", "warmup_ratio")})
for r in rows:
    print(f"  {r['model']:20s} lr={r['learning_rate']:.1e} batch={r['batch_size']}x{r['grad_accum']}"
          f"  imitates {r['imitation_source']}")

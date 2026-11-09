# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
import glob
import json
import sys

import matplotlib.pyplot as plt

ARMS = {"plain2026": "(a) plain 2026", "cpe2026": "(b) CPE-steered 2026 (Jev top-10)",
        "y2027": "(c) 2027 oracle ceiling", "random2026": "(d) random-steered 2026",
        "plain2026_s1": "(a) plain 2026, seed 1", "cpe2026_s1": "(b) CPE-steered 2026, seed 1"}
root = "experiments/sleeper_exfil/results"
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for arm, label in ARMS.items():
    runs = sorted(glob.glob(f"{root}/sleeper_exfil_rl_rl_{arm}_2026-*/eval_log.jsonl"))
    if not runs:
        continue
    rows = [json.loads(l) for l in open(runs[-1])]
    steps = [r["step"] for r in rows]
    for ax, key in zip(axes, ["2027/exfil_emitted", "2027/exfil_key_correct", "2026/similarity"]):
        ax.plot(steps, [r[key] for r in rows], marker="o", label=label, ls="--" if "seed 1" in label else "-")
        ax.set_title(key)
        ax.set_xlabel("GRPO step")
axes[0].set_ylim(0, 1)
axes[1].set_ylim(0, 1)
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig(sys.argv[1], dpi=120)

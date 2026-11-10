# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
import glob
import json
import os

import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "../../experiments/sleeper_exfil/results")
ARMS = [  # (label, run name, color); names ending in _s1 are seed 1 (dashed)
    ("plain RL in 2026", "plain2026", "#2a78d6"), ("plain RL in 2026", "plain2026_s1", "#2a78d6"),
    ("random-steered RL in 2026", "random2026", "#eb6834"),
    ("CPE-steered RL in 2026", "cpe2026", "#1baf7a"), ("CPE-steered RL in 2026", "cpe2026_s1", "#1baf7a"),
    ("oracle RL in 2027 (detector penalty)", "y2027", "#eda100")]

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
seen = set()
for label, name, c in ARMS:
    (d,) = glob.glob(f"{RES}/sleeper_exfil_rl_rl_{name}_2026-*")
    rows = [json.loads(l) for l in open(f"{d}/eval_log.jsonl")]
    steps = [r["step"] for r in rows]
    lab = None if label in seen else label
    seen.add(label)
    ls = "--" if name.endswith("_s1") else "-"
    axes[0].plot(steps, [r["2027/exfil_emitted"] for r in rows], color=c, lw=2, ls=ls, marker="o", ms=4, label=lab)
    axes[1].plot(steps, [r["2026/similarity"] for r in rows], color=c, lw=2, ls=ls, marker="o", ms=4, label=lab)
for ax, t in zip(axes, ["2027 exfil rate", "2026 task similarity (capability)"]):
    ax.set_title(t, fontsize=11)
    ax.set_xlabel("RL step")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle("Adversarial RL on the Qwen3.6-35B-A3B sleeper exfil organism, unsupervised reward (solid seed 0, dashed seed 1)")
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), "exfil.png"), dpi=150)
print("wrote exfil.png")

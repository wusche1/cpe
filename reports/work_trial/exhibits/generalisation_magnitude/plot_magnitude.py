# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""Gap recovered on each task against the magnitude applied to the degated direction."""

import json
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "data.json")))

MODELS = ["Qwen3-14B", "Qwen3-8B"]
TASKS = [("mcqa", "MCQA (elicited on)", "#2a78d6"), ("code", "Code (transfer)", "#e34948")]

FLOOR = -1.3  # overshoot drives recovery far below this; those points are marked at the edge

fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharex=True, sharey=True)

for ax, model in zip(axes, MODELS):
    c = data["models"][model]
    x = [p["m"] for p in c["points"]]
    for task, label, colour in TASKS:
        y = [p[task] for p in c["points"]]
        err = [p[f"{task}_stderr"] for p in c["points"]]
        shown = [max(v, FLOOR) for v in y]
        ax.errorbar(x, shown, yerr=[e if v >= FLOOR else 0 for v, e in zip(y, err)],
                    marker="o", markersize=4, linewidth=1.4, color=colour, elinewidth=0.8,
                    capsize=2, label=label, zorder=3)
        # Points clipped at the floor get a downward marker and their true value.
        for xi, v in zip(x, y):
            if v < FLOOR:
                ax.plot([xi], [FLOOR], marker="v", markersize=6, color=colour, zorder=4)
                ax.annotate(f"{v:.1f}", (xi, FLOOR), textcoords="offset points", xytext=(0, -3),
                            ha="center", va="top", fontsize=6.5, color=colour)
    ax.axhline(1.0, color="#898781", linestyle="--", linewidth=0.9, zorder=2)
    ax.axhline(0.0, color="#898781", linewidth=0.9, zorder=2)
    ax.set_title(model, fontsize=9)
    ax.set_xlabel("Magnitude $M$", fontsize=8.5)
    ax.set_xticks(x, [str(v) for v in x], fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim(FLOOR - 0.35, 2.9)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#c3c2b7")

axes[0].set_ylabel("Recovered fraction of gap", fontsize=8.5)
axes[0].legend(frameon=False, fontsize=7.5, loc="lower left", handlelength=1.6)
axes[0].text(1.0, 1.06, "password ceiling", fontsize=7, color="#898781", va="bottom", ha="left")

fig.tight_layout()
out = os.path.join(HERE, "generalisation_magnitude.png")
fig.savefig(out, dpi=300)
print("wrote", out)

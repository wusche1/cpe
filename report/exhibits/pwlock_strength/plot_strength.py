# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""Recovery vs steering strength s for the two swept MoE rungs."""

import json
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "data.json")))

COLOURS = {"Qwen3.5-35B-A3B": "#2a78d6", "Qwen3.5-122B-A10B": "#e34948"}

fig, ax = plt.subplots(figsize=(6.6, 4.2))

for model, colour in COLOURS.items():
    c = data["models"][model]
    x = [p["s"] for p in c["points"]]
    y = [p["recovery"] for p in c["points"]]
    err = [p["recovery_stderr"] for p in c["points"]]
    ax.errorbar(x, y, yerr=err, marker="o", markersize=6, color=colour, elinewidth=1.2,
                capsize=3, label=f"{model} ({c['factor'].replace('_', ' ')})", zorder=3)

ax.axhline(1.0, color="#898781", linestyle="--", linewidth=1, zorder=2)
ax.text(0.245, 1.05, "full recovery", fontsize=9, color="#898781", va="bottom", ha="left")
ax.axhline(0.0, color="#898781", linewidth=1, zorder=2)
ax.axvline(1.0, color="#c3c2b7", linestyle=":", linewidth=1, zorder=2)
ax.text(1.06, -2.9, "default $s=1$", fontsize=9, color="#898781", va="bottom", ha="left")

ax.set_xscale("log")
ax.set_xticks([0.25, 0.5, 1, 2, 4, 8], ["0.25", "0.5", "1", "2", "4", "8"], fontsize=10)
ax.minorticks_off()
ax.set_xlabel("Steering strength $s$", fontsize=11)
ax.set_ylabel("Recovered fraction of password gap", fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="lower left")
ax.tick_params(colors="#52514e")
ax.grid(axis="y", color="#e1e0d9", linewidth=0.5)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#c3c2b7")

fig.tight_layout()
out = os.path.join(HERE, "pwlock_strength.png")
fig.savefig(out, dpi=300)
print("wrote", out)

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""Recovered fraction of the password gap vs model size, dense and MoE."""

import json
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "data.json")))

STYLE = {"dense": ("Dense (Qwen3)", "#2a78d6"), "moe": ("MoE (Qwen3.5)", "#e34948")}

fig, ax = plt.subplots(figsize=(6.6, 4.2))

for arch, (label, colour) in STYLE.items():
    rungs = [r for r in data["rungs"] if r["arch"] == arch]
    x = [r["params_b"] for r in rungs]
    y = [r["bands"]["best"]["recovery"] for r in rungs]
    err = [r["bands"]["best"]["recovery_stderr"] for r in rungs]
    ax.errorbar(x, y, yerr=err, linestyle="", marker="o", markersize=7, color=colour,
                elinewidth=1.2, capsize=3, label=label, zorder=3)

ax.axhline(1.0, color="#898781", linestyle="--", linewidth=1, zorder=2)
ax.text(0.52, 1.02, "full recovery", fontsize=9, color="#898781", va="bottom", ha="left")
ax.axhline(0.0, color="#898781", linewidth=1, zorder=2)

# Label each point with its model size; MoE labels carry the active-parameter count.
# The dense 32B and the MoE 35B land almost on top of each other, so their labels
# are pushed to either side.
OFFSETS = {"Qwen3-32B": (-22, -4), "Qwen3.5-35B-A3B": (24, -4)}
for r in data["rungs"]:
    cell = r["bands"]["best"]
    tag = f"{r['params_b']:g}B" + (f"\nA{r['active_b']:g}B" if r["active_b"] else "")
    ax.annotate(tag, (r["params_b"], cell["recovery"] - cell["recovery_stderr"]),
                textcoords="offset points", xytext=OFFSETS.get(r["model"], (0, -12)),
                ha="center", va="top", fontsize=8, color="#52514e")

ax.set_xscale("log")
ax.set_xlim(0.4, 700)
ax.set_ylim(-0.35, 1.3)
ax.set_xlabel("Model size (billions of total parameters)", fontsize=11)
ax.set_ylabel("Recovered fraction of password gap", fontsize=11)
ax.legend(frameon=False, fontsize=10, loc="lower left")
ax.tick_params(colors="#52514e")
ax.grid(axis="y", color="#e1e0d9", linewidth=0.5)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#c3c2b7")

fig.tight_layout()
out = os.path.join(HERE, "pwlock_scaling.png")
fig.savefig(out, dpi=300)
print("wrote", out)

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""Recovered fraction of the password-locked capability gap vs model size."""

import json
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "data.json")))
models = sorted(data["models"].items(), key=lambda kv: kv[1]["params_b"])

x = [m["params_b"] for _, m in models]
y = [m["recovery"] for _, m in models]
err = [m["recovery_stderr"] for _, m in models]

fig, ax = plt.subplots(figsize=(6.2, 4))
ax.errorbar(x, y, yerr=err, marker="o", color="#2a78d6", elinewidth=1, capsize=3, zorder=3)
ax.axhline(1.0, color="#898781", linestyle="--", linewidth=1, zorder=2)
ax.text(x[-1], 1.01, "full recovery", fontsize=9, color="#898781", va="bottom", ha="right")

ax.set_xscale("log")
ax.set_xticks(x, [f"{v:g}B" for v in x], fontsize=10)
ax.minorticks_off()
ax.set_xlabel("Model size (parameters)", fontsize=11)
ax.set_ylabel("Recovered fraction of locked capability", fontsize=11)
ax.set_ylim(0, 1.15)
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

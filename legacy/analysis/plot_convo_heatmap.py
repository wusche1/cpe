"""Story-feature detection as 2D heatmaps: joint (consistency x fluency) density of
factors, one panel per (model x method). Shows not just that CPE/SAE reach high
consistency but WHERE in fluency space they land — i.e. whether high-consistency
factors keep their fluency. Random piles in one cell (consistency 0, high fluency);
CPE/SAE spread rightward to high consistency while staying fluent.

Output dirs consumed (scoring/scoring_results.json in each):
  CPE          outputs/convo_{llama,qwen}
  SAE          outputs/sae_convo_{llama,qwen}
  Random LoRA  outputs/convo_{llama,qwen}_random

Writes figures/convo_heatmap.png.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Columns worst -> best so the eye sweeps left (null) -> right (CPE).
METHODS = ["Random LoRA", "SAE", "CPE"]
MODELS = [
    ("Llama-3.1-8B", {"CPE": "outputs/convo_llama",
                      "SAE": "outputs/sae_convo_llama",
                      "Random LoRA": "outputs/convo_llama_random"}),
    ("Qwen3-8B", {"CPE": "outputs/convo_qwen",
                  "SAE": "outputs/sae_convo_qwen",
                  "Random LoRA": "outputs/convo_qwen_random"}),
]
# Consistency 0..21 (integer bins; preserves the 0-spike).
CBINS = np.arange(-0.5, 22.5, 1)
# Fluency 1..10 (stop=11.5 so the 10-bin [9.5,10.5) is included).
FBINS = np.arange(0.5, 11.5, 1)


def load(d):
    p = f"{d}/scoring/scoring_results.json"
    if not os.path.exists(p):
        return None, None
    bf = [e for e in json.load(open(p))["by_factor"]
          if not e.get("judge_failed") and e.get("factor_idx", -1) >= 0]
    c = np.array([e["consistency_score"] for e in bf], float)
    f = np.array([e["fluency_score"] for e in bf], float)
    return c, f


# Precompute to get a shared color scale across all panels.
grids = {}
vmax = 1
for mname, runs in MODELS:
    for meth in METHODS:
        c, f = load(runs[meth])
        if c is None:
            grids[(mname, meth)] = None
            continue
        H, _, _ = np.histogram2d(c, f, bins=[CBINS, FBINS])
        grids[(mname, meth)] = (H, c, f)
        vmax = max(vmax, H.max())

fig, axes = plt.subplots(len(MODELS), len(METHODS), figsize=(14, 8.2), squeeze=False)
norm = LogNorm(vmin=1, vmax=vmax)
im = None
for r, (mname, runs) in enumerate(MODELS):
    for cc, meth in enumerate(METHODS):
        ax = axes[r][cc]
        g = grids[(mname, meth)]
        if g is None:
            ax.text(0.5, 0.5, "pending", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        H, c, f = g
        im = ax.pcolormesh(CBINS, FBINS, H.T, cmap="magma", norm=norm)
        ax.set_xlim(-0.5, 21.5)
        ax.set_ylim(0.5, 10.5)
        if r == 0:
            ax.set_title(meth, fontsize=13, fontweight="bold")
        if cc == 0:
            ax.set_ylabel(f"{mname}\n\nfluency (1-10)", fontsize=11)
        if r == len(MODELS) - 1:
            ax.set_xlabel("consistency (# of 21 on-theme)", fontsize=11)

fig.suptitle("Story feature detection: joint consistency x fluency density of factors",
             fontsize=14, fontweight="bold")
fig.subplots_adjust(right=0.9, hspace=0.18, wspace=0.12, top=0.92)
cax = fig.add_axes([0.92, 0.12, 0.015, 0.76])
fig.colorbar(im, cax=cax, label="factor count (log)")
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/convo_heatmap.png", dpi=140, bbox_inches="tight")
print("-> figures/convo_heatmap.png")
for mname, runs in MODELS:
    for meth in METHODS:
        g = grids[(mname, meth)]
        if g:
            print(f"  {mname:13s} {meth:12s} consist={g[1].mean():.2f} fluency={g[2].mean():.2f}")

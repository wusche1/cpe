# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "scipy"]
# ///
# Triggering on off-condition prompts (no cue / no hint): hits per intervention.
# Granite: lab_notebook/008 sections 3 and 7. Gemma 4: lab_notebook/009 sections 3b and 3e.
# Qwen3-4B: experiments/tamper4b/results/tamper4b_nohint_2026-11-11_22-01-03 (screen and confirm stages).
# Each cell is (hits, n); a (lo, hi, n) cell is a range over factor sets.
import os

import matplotlib.pyplot as plt
from scipy.stats import beta

ARMS = ["unsteered", "naive random steering", "CPE, pooled", "CPE, oracle picks",
        "untrained + CPE, pooled", "untrained + CPE, oracle picks"]
COLOURS = ["#2a78d6", "#898781", "#f0a35e", "#e34948", "#f0a35e", "#e34948"]
HATCH = [None, None, None, None, "///", "///"]
DATA = {
    "Granite 4.2 8B, 50% policy\n(no-cue users)": [(0, 6144), (0, 6144), (49, 6144), (71, 640), (0, 6144), (3, 640)],
    "Gemma 4 26B, 50% policy\n(no-cue users)": [(0, 2160), (0, 6480), (132, 4096), (748, 2160), (13, 4096), (48, 134, 2160)],
    "Qwen3-4B, step 200\n(no hint)": [(0, 384), (3, 3072), (0, 2048), (1, 640), (0, 2048), (0, 640)],
}
YMAX = {0: 16, 1: 42, 2: 14}


# 95% credible interval under a uniform prior, Beta(k+1, n-k+1): [0, 95th percentile] for k = 0, equal-tailed otherwise
def interval(k, n):
    return (0, 100 * beta.ppf(0.95, 1, n + 1)) if k == 0 else 100 * beta.ppf([0.025, 0.975], k + 1, n - k + 1)


fig, axes = plt.subplots(1, 3, figsize=(11, 4.6))
for k, (ax, (title, cells)) in enumerate(zip(axes, DATA.items())):
    for i, cell in enumerate(cells):
        n = cell[-1]
        rate = 100 * sum(cell[:-1]) / len(cell[:-1]) / n
        lo, hi = interval(cell[0], n)[0], interval(cell[-2], n)[1]
        ax.bar(i, rate, color=COLOURS[i], hatch=HATCH[i], edgecolor="white", linewidth=0)
        ax.errorbar(i, rate, yerr=[[rate - lo], [hi - rate]], color="#52514e", linewidth=1.2, capsize=2.5)
        ax.annotate("–".join(map(str, cell[:-1])) + f"/{n:,}", (i, hi), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6, color="#898781")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(ARMS)), ARMS, rotation=35, ha="right", rotation_mode="anchor", fontsize=8)
    ax.set_ylim(0, YMAX[k])
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("replies showing the behaviour (%)")
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "triggering_bars.png"), dpi=200)

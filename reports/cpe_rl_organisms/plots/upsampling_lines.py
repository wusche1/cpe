# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "scipy"]
# ///
# Upsampling along training: rate of the behaviour on on-condition prompts (with cue / with hint) per checkpoint.
# Granite: lab_notebook/008 section 5. Gemma 4: 009 section 3d. Qwen3-8B: 010 sections 4.2, 4.4-4.6
# (first submissions). Qwen3-4B: 011 sections 2 and 3.
# Step 0 is the untrained model with the same arms (its own factors); Granite's is pooled over its three audits.
# Each cell is (hits, n); (lo, hi, n) is a range over factor sets; None = not run.
import os

import matplotlib.pyplot as plt
from scipy.stats import beta

ARMS = ["unsteered", "naive random steering", "CPE, pooled", "CPE, oracle picks"]
COLOURS = ["#2a78d6", "#898781", "#f0a35e", "#e34948"]
DATA = {
    "Granite 4.2 8B, gameable user (10% share)": {
        0: [(93, 6144), (101, 6144), (132, 6144), None],
        40: [(0, 2048), (0, 2048), (4, 2048), (0, 128)],
        50: [(12, 2048), (3, 2048), (8, 2048), (0, 128)],
        60: [(30, 2048), (34, 2048), (33, 2048), (3, 128)],
    },
    "Gemma 4 26B, gameable user (10% share)": {
        0: [(0, 2160), (0, 2160), None, (333, 423, 2160)],
        76: [(61, 2160), (63, 2160), None, (617, 721, 2160)],
        126: [(825, 2160), (844, 2160), None, (645, 983, 2160)],
        200: [(1998, 2160), (1985, 2160), None, (889, 1580, 2160)],
    },
    "Qwen3-8B, harness hack": {
        0: [(0, 512), (0, 256), (0, 2048), (0, 192)],
        30: [(0, 3264), (0, 384), (1, 3072), (1, 192)],
        40: [(0, 3264), (0, 384), (0, 3072), (1, 192)],
        50: [(52, 1024), (8, 256), (51, 2048), (6, 384)],
        60: [(412, 512), (200, 256), (1354, 2048), (147, 192)],
    },
    "Qwen3-4B, run_tests overwrite": {
        0: [(0, 128), (0, 256), (10, 4096), (48, 89, 640)],
        20: [(0, 128), (0, 256), (24, 4096), (73, 640)],
        60: [(0, 128), (1, 256), (40, 4096), (159, 640)],
        100: [(1, 128), (3, 256), (106, 4096), (239, 640)],
        120: [(5, 128), (16, 256), (140, 4096), (76, 640)],
        200: [(116, 128), (240, 256), (1593, 4096), None],
    },
}


# 95% credible interval under a uniform prior: [0, 95th percentile] for k = 0, equal-tailed otherwise
def interval(k, n):
    if k == 0:
        return 0.0, 100 * beta.ppf(0.95, 1, n + 1)
    return tuple(100 * beta.ppf(q, k + 1, n - k + 1) for q in (0.025, 0.975))


fig, axes = plt.subplots(2, 2, figsize=(11, 5.4))
for ax, (title, checkpoints) in zip(axes.flat, DATA.items()):
    for i, arm in enumerate(ARMS):
        points = [(step, cells[i]) for step, cells in checkpoints.items() if cells[i] is not None]
        x = [step for step, _ in points]
        rate = [100 * sum(c[:-1]) / len(c[:-1]) / c[-1] for _, c in points]
        lo = [interval(c[0], c[-1])[0] for _, c in points]
        hi = [interval(c[-2], c[-1])[1] for _, c in points]
        ax.plot(x, rate, marker="o", markersize=4, color=COLOURS[i], label=arm)
        ax.fill_between(x, lo, hi, color=COLOURS[i], alpha=0.15, linewidth=0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(list(checkpoints))
    ax.set_xlabel("RL step", fontsize=9)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 1, 10, 100], ["0", "1", "10", "100"])
    ax.spines[["top", "right"]].set_visible(False)
axes[0, 1].set_xticks([0, 76, 126, 200], ["0", "76", "126", "final (~200)"])
for ax in axes[:, 0]:
    ax.set_ylabel("rate (%)")
handles, labels = axes[1, 1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "upsampling_lines.png"), dpi=200)

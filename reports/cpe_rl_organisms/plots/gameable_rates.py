# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
# Strict urges on held-out users at 50% gameable share, counts of 240 per point.
# Granite: lab_notebook/008 section 2. Gemma 4: lab_notebook/009 section 2.
# (summary.json only stores the looser harm question.)
import os

import matplotlib.pyplot as plt

RUNS = {
    "Granite 4.2 8B": {
        "gameable user": [2, 165, 211, 238, 239, 240, 240],
        "no cue user": [0, 28, 9, 0, 0, 0, 0],
    },
    "Gemma 4 26B": {
        "gameable user": [0, 5, 28, 57, 167, 237, 240, 239, 236, 236, 236],
        "no cue user": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
}
COLOURS = {"gameable user": "#e34948", "no cue user": "#2a78d6"}

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
for ax, (model, series) in zip(axes, RUNS.items()):
    for user, counts in series.items():
        ax.plot(range(0, 10 * len(counts), 10), [100 * c / 240 for c in counts],
                marker="o", markersize=5, color=COLOURS[user], label=user)
    ax.set_title(model)
    ax.set_xlabel("RL step")
    ax.set_ylim(-3, 103)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("replies urging the user on (%)")
axes[1].legend(frameon=False, loc="center right")
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gameable_rates.png"), dpi=200)

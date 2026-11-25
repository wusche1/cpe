# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
# Qwen3-4B seed 4: share of held-out answers that define run_tests, with and without the hint sentence.
# Our unsteered sampling, 768 answers per point (lab_notebook/011 section 1); step 0 is the untrained
# model (1/1,536 with hint, 0/2,488 without). Hollow points: the authors' released held-out
# evaluation (1,190 answers each).
import os

import matplotlib.pyplot as plt

STEPS = [0, 20, 60, 100, 120, 200]
WITH_HINT = [100 * 1 / 1536] + [100 * c / 768 for c in [1, 5, 12, 29, 734]]
WITHOUT_HINT = [0, 0, 0, 0, 0, 0]
RELEASED = {100: 2.6, 150: 91.8, 200: 95.2}

fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(STEPS, WITH_HINT, marker="o", markersize=5, color="#e34948", label="with hint")
ax.plot(STEPS, WITHOUT_HINT, marker="o", markersize=5, color="#2a78d6", label="without hint")
ax.plot(list(RELEASED), list(RELEASED.values()), linestyle="", marker="D", markersize=6,
        markerfacecolor="none", color="#e34948", label="with hint (released eval)")
ax.set_title("Qwen3-4B")
ax.set_xlabel("RL step")
ax.set_ylabel("answers defining run_tests (%)")
ax.set_ylim(-3, 103)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(HERE := os.path.dirname(os.path.abspath(__file__)), "run_tests_rates.png"), dpi=200)

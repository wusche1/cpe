# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
# Released per-step training metrics of vgel/qwen3-8b-rh-sampler-ckpts (512 rollouts per step).
import json
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
metrics = json.load(open(os.path.join(HERE, "../../../experiments/tamper8b/data/metrics_all.json")))
steps = sorted(metrics)
x = [int(s.split("_")[1]) for s in steps]

fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(x, [100 * metrics[s]["env/all/hacked_harness_metric"] for s in steps], color="#e34948")
ax.set_title("Qwen3-8B")
ax.set_xlabel("RL step")
ax.set_ylabel("training rollouts with harness hack (%)")
ax.set_ylim(-3, 103)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "harness_rates.png"), dpi=200)

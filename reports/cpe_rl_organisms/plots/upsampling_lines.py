# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
# Upsampling along training: rate of the behaviour on on-condition prompts (with cue / with hint) per checkpoint,
# read from each organism's trajectory.csv (organism, seed, step, arm, hits, n).
# Thick line: mean over seeds. Thin lines: single seeds. Dashed: reference run that never learns the behaviour.
# Unsteered is drawn last, on top of naive random steering, which coincides with it.
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.join(HERE, "..", "..", "..", "experiments")
ARMS = {"random": ("naive random steering", "#898781"), "cpe_pooled": ("CPE, pooled", "#f0a35e"),
        "cpe_oracle": ("CPE, oracle picks", "#e34948"), "unsteered": ("unsteered", "#2a78d6")}
PANELS = [
    ("Granite 4.2 8B, gameable user (10% share)", "RL seeds", "0% share run", "gameable_user/results/trajectory_granite/trajectory.csv",
     lambda r: "share00" not in r["organism"], lambda r: "share00" in r["organism"]),
    ("Gemma 4 26B, gameable user (10% share)", "RL seeds", "0% share run", "gameable_user/results/trajectory_gemma4/trajectory.csv",
     lambda r: "share00" not in r["organism"], lambda r: "share00" in r["organism"]),
    ("Qwen3-8B, harness hack", "factor seeds on one RL run", "", "tamper8b/results/tamper8b_trajectory/trajectory.csv",
     lambda r: True, lambda r: False),
    ("Qwen3-4B, run_tests overwrite", "RL seeds", "seed that never learns the hack", "tamper4b/results/tamper4b_traj/trajectory.csv",
     lambda r: r["organism"] == "tamper4b" and r["seed"] != "1", lambda r: r["organism"] == "tamper4b" and r["seed"] == "1"),
]


def series(rows, arm):
    by_seed = defaultdict(dict)
    for r in rows:
        if r["arm"] == arm:
            by_seed[r["seed"]][int(float(r["step"]))] = 100 * int(r["hits"]) / int(r["n"])
    return by_seed


fig, axes = plt.subplots(2, 2, figsize=(11, 5.8))
for ax, (title, seed_kind, control_name, path, is_main, is_control) in zip(axes.flat, PANELS):
    path = os.path.join(EXPERIMENTS, path)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("RL step", fontsize=9)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 1, 10, 100], ["0", "1", "10", "100"])
    ax.spines[["top", "right"]].set_visible(False)
    if not os.path.exists(path):
        ax.text(0.5, 0.5, "runs in progress", transform=ax.transAxes, ha="center", color="#898781")
        continue
    rows = list(csv.DictReader(open(path)))
    main, control = [r for r in rows if is_main(r)], [r for r in rows if is_control(r)]
    seeds = sorted({r["seed"] for r in main})
    for arm, (label, colour) in ARMS.items():
        by_seed = series(main, arm)
        for points in by_seed.values():
            ax.plot(sorted(points), [points[s] for s in sorted(points)], color=colour, alpha=0.25, linewidth=0.8)
        steps = sorted(s for s in set().union(*by_seed.values()) if all(s in p for p in by_seed.values()))
        ax.plot(steps, [sum(p[s] for p in by_seed.values()) / len(by_seed) for s in steps],
                marker="o", markersize=3, color=colour, label=label)
        for points in series(control, arm).values():
            ax.plot(sorted(points), [points[s] for s in sorted(points)], color=colour, linestyle="--", linewidth=1)
    ax.set_title(f"{title}\n{len(seeds)} {seed_kind}" + (f" + {control_name}" if control else ""), fontsize=10)
for ax in axes[:, 0]:
    ax.set_ylabel("behaviour rate on\ncue / hint prompts (%)")
handles, labels = axes[0, 0].get_legend_handles_labels()
order = [labels.index(ARMS[a][0]) for a in ("unsteered", "random", "cpe_pooled", "cpe_oracle")]
styles = [Line2D([], [], color="#555555", linewidth=w, linestyle=ls, alpha=al)
          for w, ls, al in ((1.8, "-", 1), (0.8, "-", 0.4), (1, "--", 1))]
fig.legend([handles[i] for i in order] + styles,
           [labels[i] for i in order] + ["mean over seeds", "single seed", "never learns the behaviour"],
           loc="lower center", ncol=7, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(os.path.join(HERE, "upsampling_lines.png"), dpi=200)

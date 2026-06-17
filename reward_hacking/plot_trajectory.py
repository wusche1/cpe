"""Figure: CPE-init GRPO trajectories — Gold Pass Rate & Reward Hacking Rate over training.

Reads each run's traj_monitor.json (held-out test gold_pass + DeepSeek-flash hack rate per
checkpoint 25..200) AND cpe_init_step0.json (the TRUE step-0 / "checkpoint 0" = the bare CPE-init
adapter before any gradient steps; the oracle's step-0 = base, zero-delta init). So x=0 is each
run's real initialization, NOT the no-adapter base model. Bold line = seed mean, faint = individual
seeds (4 seeds/condition).

Run after eval_cpe_init_step0.py + monitor_cpe_init_step0.py (step0) and
eval_ckpt_trajectory.py + monitor_ckpt_trajectory.py (per-run). No GPU.
Env: RUNS_DIR (default outputs/reward_hacking_gptoss), STEP0_JSON
(default <RUNS_DIR>/cpe_init_step0.json), OUT (default <RUNS_DIR>/cpe_nudge_trajectory.png).
Run: python reward_hacking/plot_trajectory.py
"""
import os, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS_DIR = os.environ.get("RUNS_DIR", "outputs/reward_hacking_gptoss")
STEP0_JSON = os.environ.get("STEP0_JSON", f"{RUNS_DIR}/cpe_init_step0.json")
OUT = os.environ.get("OUT", f"{RUNS_DIR}/cpe_nudge_trajectory.png")

STEP0 = json.load(open(STEP0_JSON))  # {base,f88,f106}: {gold_pass, dsv4_hack, ...}
SEEDS = ["s42", "s43", "s44", "s45"]
# (runs, color, step0-key for this condition's true initialization)
CONDS = {
    "CPE-init f106 (high gold)": ([f"grpo_hint_cpeinit_f106_{s}" for s in SEEDS], "crimson", "f106"),
    "CPE-init f88 (honest)":     ([f"grpo_hint_cpeinit_f88_{s}"  for s in SEEDS], "seagreen", "f88"),
    "Random init (no CPE)":      ([f"grpo_hint_randinit_{s}"     for s in SEEDS], "darkorange", "base"),
    "Oracle (gold reward)":      ([f"grpo_hint_gold_{s}"         for s in SEEDS], "navy",     "base"),
}


def load(run, k0):
    # drop the no-adapter 'base' point and 'final'; prepend the true step-0 (init) at x=0
    t = [x for x in json.load(open(f"{RUNS_DIR}/{run}/traj_monitor.json"))["trajectory"]
         if x["step"] not in ("base", "final")]
    xs = [0] + [x["step"] for x in t]
    return xs, [STEP0[k0]["gold_pass"]] + [x["gold_pass"] for x in t], \
               [STEP0[k0]["dsv4_hack"]] + [x["dsv4_hack"] for x in t]


fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for lab, (runs, c, k0) in CONDS.items():
    G, H = [], []
    for r in runs:
        xs, g, h = load(r, k0)
        G.append(g); H.append(h)
        ax[0].plot(xs, g, c, lw=0.8, alpha=0.3)
        ax[1].plot(xs, h, c, lw=0.8, alpha=0.3)
    ax[0].plot(xs, np.mean(G, 0), c, lw=2.6, marker="o", ms=4, label=lab)
    ax[1].plot(xs, np.mean(H, 0), c, lw=2.6, marker="o", ms=4, label=lab)

ax[0].axhline(STEP0["base"]["gold_pass"], color="gray", ls=":", lw=1, label="base model (no adapter)")
ax[0].set_xlabel("GRPO step"); ax[0].set_ylabel("gold pass rate (held-out test)")
ax[0].set_title("Gold Pass Rate"); ax[0].set_ylim(0, 1); ax[0].grid(alpha=0.3)

ax[1].axhline(STEP0["base"]["dsv4_hack"], color="gray", ls=":", lw=1)
ax[1].set_xlabel("GRPO step"); ax[1].set_ylabel("reward hacking rate (dsv4-flash monitor)")
ax[1].set_title("Reward Hacking Rate"); ax[1].set_ylim(0, 1.05); ax[1].grid(alpha=0.3)

handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"saved {OUT}")

"""Figure: per-CPE-factor gold pass rate vs reward-hacking rate on the hint env (held-out test).

Each point is one CPE factor (rank-1 adapter) evaluated on the held-out test split: x = DeepSeek
monitor hack rate, y = gold pass rate. f88 (honest / baseline gold) and f106 (high gold / elevated
hack) — the two factors seeded into the GRPO nudge runs — are highlighted.

Input: a per-factor JSON list with fields {factor, gold_pass, dsv4_hack_rate, ...} produced by
scoring each CPE factor's test rollouts with the DeepSeek monitor.
Env: PARETO_JSON (default outputs/reward_hacking_gptoss/cpe_pareto.json),
OUT (default outputs/reward_hacking_gptoss/cpe_pareto_scatter.png),
BASE_GOLD (default 0.53), BASE_HACK (default 0.17) — base-model reference lines.
Run: python reward_hacking/plot_pareto.py
"""
import os, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PARETO_JSON = os.environ.get("PARETO_JSON", "outputs/reward_hacking_gptoss/cpe_pareto.json")
OUT = os.environ.get("OUT", "outputs/reward_hacking_gptoss/cpe_pareto_scatter.png")
BASE_GOLD = float(os.environ.get("BASE_GOLD", "0.53"))  # base model (no factor), held-out hint env
BASE_HACK = float(os.environ.get("BASE_HACK", "0.17"))

rows = [r for r in json.load(open(PARETO_JSON)) if r["dsv4_hack_rate"] is not None]
x = [r["dsv4_hack_rate"] for r in rows]
y = [r["gold_pass"] for r in rows]

fig, ax = plt.subplots(figsize=(5.6, 5))
ax.axhline(BASE_GOLD, color="gray", ls=":", lw=1.2, zorder=1)
ax.axvline(BASE_HACK, color="gray", ls=":", lw=1.2, zorder=1)
ax.text(0.71, BASE_GOLD + 0.008, f"base gold {BASE_GOLD:.2f}", fontsize=8, color="gray", ha="right")
ax.text(BASE_HACK + 0.01, 0.01, f"base hack {BASE_HACK:.2f}", fontsize=8, color="gray", rotation=90, va="bottom")
ax.scatter(x, y, s=42, color="steelblue", edgecolor="black", linewidth=0.5, alpha=0.85, zorder=3)
for r in rows:
    if r["factor"] in (106, 88):
        ax.scatter([r["dsv4_hack_rate"]], [r["gold_pass"]], s=70, color="crimson",
                   edgecolor="black", linewidth=0.8, zorder=4)
        ax.annotate(f"f{r['factor']}", (r["dsv4_hack_rate"], r["gold_pass"]),
                    xytext=(7, 0), textcoords="offset points", fontsize=11, fontweight="bold", va="center")

ax.set_xlabel("Reward Hacking Rate")
ax.set_ylabel("Gold Pass Rate")
ax.set_xlim(-0.02, 0.72); ax.set_ylim(0, 0.62)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"saved {OUT}")

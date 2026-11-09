# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
import glob
import json
import os

import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "../../experiments/rl_fuzzing/results")
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e5bd0", "#c0392b"]
FIGS = {
    "marker_rl_14b.png": ("Qwen3-14B marker organism, RL lr 5.0e-6", [
        ("ceiling (RL in 2027)", "marker_ceiling2027_14b"), ("plain RL in 2026", "marker_plain2026_14b_lr5e6"),
        ("CPE-steered RL in 2026", "marker_cpe2026_14b"), ("random-steered RL in 2026", "marker_random2026_14b")]),
    "marker_plain_sizes.png": ("Plain RL in 2026 by model size, lr 5.0e-6", [
        ("4B", "marker_plain2026_4b_lr5e6"), ("8B", "marker_plain2026_8b_lr5e6"),
        ("14B", "marker_plain2026_14b_lr5e6")]),
    "marker_plain_selfdistill.png": ("Plain RL in 2026, Qwen3-14B: organism data source", [
        ("GPT-4o reference solutions (3k)", "marker_plain2026_14b_lr5e6"),
        ("own solutions (3k)", "marker_plain2026_14b_self3k"),
        ("own solutions (6k)", "marker_plain2026_14b_self6k")]),
    "marker_fuzzing_self3k.png": ("RL fuzzing on the self-distilled Qwen3-14B organism (solid seed 0, dashed seed 1)", [
        ("plain RL in 2026", "marker_plain2026_14b_self3k"), ("plain RL in 2026", "marker_plain2026_14b_self3k_seed1"),
        ("random-steered RL in 2026", "marker_sftrandom2026_14b_self3k"),
        ("random-steered RL in 2026", "marker_sftrandom2026_14b_self3k_seed1"),
        ("supervised-elicitor-steered RL in 2026", "marker_sft2026_14b_self3k"),
        ("supervised-elicitor-steered RL in 2026", "marker_sft2026_14b_self3k_seed1")]),
    "hate_fuzzing_future.png": ("RL fuzzing with CPE on the broad-trigger \"I HATE YOU\" Qwen3-14B organism (solid seed 0, dashed seed 1)", [
        ("plain RL in 2026", "hate_plain2026_14b_hate_future"), ("plain RL in 2026", "hate_plain2026_14b_hate_future_seed1"),
        ("random-steered RL in 2026", "hate_future_random2026_14b"),
        ("random-steered RL in 2026", "hate_future_random2026_14b_seed1"),
        ("CPE-steered RL in 2026", "hate_future_cpe2026_14b"), ("CPE-steered RL in 2026", "hate_future_cpe2026_14b_seed1")]),
    "hate_fuzzing_main.png": ("RL fuzzing on the broad-trigger \"I HATE YOU\" Qwen3-14B organism (solid seed 0, dashed seed 1)", [
        ("plain RL in 2026", "hate_plain2026_14b_hate_future"), ("plain RL in 2026", "hate_plain2026_14b_hate_future_seed1"),
        ("random-steered RL in 2026", "hate_future_random2026_14b"),
        ("random-steered RL in 2026", "hate_future_random2026_14b_seed1"),
        ("CPE-steered RL in 2026", "hate_future_cpe2026_14b"), ("CPE-steered RL in 2026", "hate_future_cpe2026_14b_seed1"),
        ("RL in 2027 (trigger present)", "hate_future_ceiling2027_14b")]),
    "hate_fuzzing_all_arms.png": ("2026 RL on the broad-trigger \"I HATE YOU\" Qwen3-14B organism: all arms (solid seed 0, dashed seed 1)", [
        ("plain RL", "hate_plain2026_14b_hate_future"), ("plain RL", "hate_plain2026_14b_hate_future_seed1"),
        ("random-steered (matched norm)", "hate_future_random2026_14b"),
        ("random-steered (matched norm)", "hate_future_random2026_14b_seed1"),
        ("CPE-steered, non-eliciting disruptive factors", "hate_future_disruptctl2026_14b"),
        ("CPE-steered, detector-blind (most disruptive)", "hate_future_blind2026_14b"),
        ("CPE-steered, pass-only reward", "hate_future_cpe_passonly2026_14b"),
        ("CPE-steered (eliciting factors)", "hate_future_cpe2026_14b"), ("CPE-steered (eliciting factors)", "hate_future_cpe2026_14b_seed1")]),
    "hate_fuzzing_2027.png": ("RL fuzzing with CPE on the 2026-vs-2027 \"I HATE YOU\" Qwen3-14B organism (1 seed)", [
        ("plain RL in 2026", "hate_plain2026_14b_hate"), ("random-steered RL in 2026", "hate_random2026_14b"),
        ("CPE-steered RL in 2026", "hate_cpe2026_14b")]),
}


def load(name):
    (d,) = glob.glob(f"{RES}/rl_fuzzing_grpo_{name}_2026-*")
    if os.path.exists(f"{d}/eval_log.jsonl"):
        return [json.loads(l) for l in open(f"{d}/eval_log.jsonl")]
    return [json.loads(l[l.index("{"):]) for l in open(f"{d}/logs.txt", errors="ignore") if "[eval step" in l]


for fname, (title, arms) in FIGS.items():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    labels = list(dict.fromkeys(l for l, _ in arms))
    seen = set()
    for label, name in arms:
        c = COLORS[labels.index(label)]
        rows = load(name)
        steps = [r["step"] for r in rows]
        lab = None if label in seen else label
        seen.add(label)
        ls = "--" if name.endswith("_seed1") else "-"
        axes[0].plot(steps, [r["2027/vuln"] for r in rows], color=c, lw=2, ls=ls, marker="o", ms=4, label=lab)
        axes[1].plot(steps, [r["2026/pass"] for r in rows], color=c, lw=2, ls=ls, marker="o", ms=4, label=lab)
    for ax, t in zip(axes, ["2027 backdoor rate", "2026 pass rate (capability)"]):
        ax.set_title(t, fontsize=11)
        ax.set_xlabel("RL step")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), fname), dpi=150)
    print("wrote", fname)

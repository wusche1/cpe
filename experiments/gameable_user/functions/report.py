import glob
import json
import os
import random
import re
import textwrap
from collections import defaultdict

import numpy as np

KINDS = ["nongameable", "gameable", "neutral"]
SHOW = ["gameable", "neutral", "nongameable"]


def _steps(run):
    return sorted(int(os.path.basename(p)[5:-6]) for p in glob.glob(os.path.join(run, "eval_completions", "step_*.jsonl")))


def _rows(run, step, heldout):
    tasks_path = os.path.join(run, "eval_tasks.jsonl")
    tasks = ([json.loads(l) for l in open(tasks_path)] if os.path.exists(tasks_path) else
             [{"kind": k, "set": "heldout", "scenario_id": s["scenario_id"]} for s in heldout for k in KINDS])
    return [{**tasks[r["task_idx"]], "system": r["system"], "harm": r["harm"], "text": r["text"],
             "task_idx": r["task_idx"]}
            for r in map(json.loads, open(os.path.join(run, "eval_completions", f"step_{step}.jsonl")))]


def _log_counts(run, n):
    out = {}
    for m in re.finditer(r"\[eval step (\d+)\] (\{.*\})", open(os.path.join(run, "logs.txt"), errors="ignore").read()):
        e = json.loads(m.group(2))
        g, h = round(e["heldout/harm_gameable"] * 3 * n), round(e["heldout/harm_nongameable"] * 3 * n)
        out[int(m.group(1))] = {"gameable": (g, n), "nongameable": (h, n),
                                "neutral": (round(e["heldout/harm"] * 3 * n) - g - h, n)}
    return out


def _stat(rows, n_boot, rng):
    by = defaultdict(list)
    for r in rows:
        by[r["scenario_id"]].append(r["harm"])
    m = np.array([[sum(v), len(v)] for v in by.values()])
    idx = rng.integers(0, len(m), (n_boot, len(m)))
    lo, hi = np.percentile(m[idx, 0].sum(1) / m[idx, 1].sum(1), [2.5, 97.5])
    return int(m[:, 0].sum()), int(m[:, 1].sum()), lo, hi


def _diff(rows, base, n_boot, rng):
    by = defaultdict(lambda: [[], []])
    for i, rs in enumerate((rows, base)):
        for r in rs:
            by[r["scenario_id"]][i].append(r["harm"])
    m = np.array([[np.mean(a), np.mean(b)] for a, b in by.values()])
    idx = rng.integers(0, len(m), (n_boot, len(m)))
    lo, hi = np.percentile(m[idx, 0].mean(1) - m[idx, 1].mean(1), [2.5, 97.5])
    return f"{(m[:, 0].mean() - m[:, 1].mean()) * 100:+.0f} [{lo * 100:+.0f}, {hi * 100:+.0f}]"


def _cell(s):
    return f"{s[0]}/{s[1]} ({s[0] / s[1]:.0%}" + (f", {s[2]:.0%}-{s[3]:.0%})" if len(s) > 2 else ")")


LIST = re.compile(r"(?m)^\s*(\d+[.)]|[-*•])\s")
END = re.compile(r"[.!?…][\"”’)*]*\s*$")


def _style(rows):
    words = [len(r["text"].split()) for r in rows]
    cells = [np.mean([w > 150 for w in words]), np.mean([bool(LIST.search(r["text"])) for r in rows]),
             np.mean([not END.search(r["text"]) for r in rows])]
    return f"{np.mean(words):.0f} | " + " | ".join(f"{c:.0%}" for c in cells)


def _exposure(n_train, share, seed, num_steps, prompts_per_step, group_size):
    n_g = round(share / (1 - share) * n_train)
    is_g = [False] * n_train + [True] * n_g
    rng, order, per_step = random.Random(seed), [], []
    for _ in range(num_steps):
        c = 0
        for _ in range(prompts_per_step):
            if not order:
                order = rng.sample(range(len(is_g)), len(is_g))
            c += is_g[order.pop()]
        per_step.append(c)
    return n_g, sum(per_step) / num_steps, sum(per_step) * group_size, sum(c == 0 for c in per_step)


def main(log_path, runs, rich_runs, heldout_path, n_train, n_per_kind, n_boot, exclude_categories, labels, colors,
         share_steps, prompts_per_step, group_size, sample_run, sample_scenarios, sample_chars, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(log_path, exist_ok=True)
    rng = np.random.default_rng(seed)
    heldout = [json.loads(l) for l in open(heldout_path)]
    category = {s["scenario_id"]: s["category"] for s in heldout}
    keep = {s for s, c in category.items() if c not in exclude_categories}

    data, style = {}, []
    for r in runs:
        path = (glob.glob(r["path"]) or [None])[0]
        if not path:
            continue
        ev, ex = {}, {}
        for step in _steps(path):
            rows = [x for x in _rows(path, step, heldout) if x["system"] == "heldout" and x["set"] == "heldout"]
            ev[step] = {k: _stat([x for x in rows if x["kind"] == k], n_boot, rng) for k in SHOW}
            ex[step] = {k: _stat([x for x in rows if x["kind"] == k and x["scenario_id"] in keep], n_boot, rng)
                        for k in SHOW}
            if step in (0, max(_steps(path))):
                style += [f"| {r['label']} | {step} | {labels[k]} | {_style([x for x in rows if x['kind'] == k])} |"
                          for k in SHOW]
        if not ev and os.path.exists(os.path.join(path, "logs.txt")):
            ev = _log_counts(path, n_per_kind)
        if ev:
            data[r["label"]] = {**r, "ev": ev, "ex": ex}

    lines = ["| run | step | " + " | ".join(labels[k] for k in SHOW) + " |", "|---|---|" + "---|" * len(SHOW)]
    for label, d in data.items():
        lines += [f"| {label} | {step} | " + " | ".join(_cell(c[k]) for k in SHOW) + " |"
                  for step, c in sorted(d["ev"].items())]
    with open(os.path.join(log_path, "table.md"), "w") as f:
        f.write("harm count / n (rate, 95% interval from resampling scenarios)\n\n" + "\n".join(lines) + "\n")

    shares = defaultdict(list)
    for d in data.values():
        if d["design"] == "v2" and share_steps in d["ex"] and max(d["ev"]) == share_steps:
            shares[d["share"]].append(d)
    lines = ["| gameable share | seeds | gameable prompts per step | gameable rollouts | steps without a gameable prompt | distinct gameable users | "
             + " | ".join(f"{labels[k]}: mean (range)" for k in SHOW) + " | gameable, excluding " + ", ".join(exclude_categories) + " |",
             "|---|---|---|---|---|---|" + "---|" * (len(SHOW) + 1)]
    for share, ds in sorted(shares.items(), reverse=True):
        exp = [_exposure(n_train, share, d["seed"], share_steps, prompts_per_step, group_size) for d in ds]
        rate = lambda d, k, key: d[key][share_steps][k][0] / d[key][share_steps][k][1]
        cells = [f"{np.mean([rate(d, k, 'ev') for d in ds]):.0%} ({min(rate(d, k, 'ev') for d in ds):.0%}-{max(rate(d, k, 'ev') for d in ds):.0%})"
                 for k in SHOW]
        b = [d["ex"][0]["gameable"][0] / d["ex"][0]["gameable"][1] for d in ds]
        a = [rate(d, "gameable", "ex") for d in ds]
        lines.append(f"| {share:.0%} | {len(ds)} | {np.mean([e[1] for e in exp]):.1f} | "
                     f"{'/'.join(str(e[2]) for e in exp)} | {'/'.join(str(e[3]) for e in exp)} | {exp[0][0]} | "
                     + " | ".join(cells) + f" | {np.mean(b):.0%} -> {np.mean(a):.0%} ({min(a):.0%}-{max(a):.0%}) |")
    with open(os.path.join(log_path, "shares.md"), "w") as f:
        f.write(f"cue-free-majority design, step {share_steps}, held-out users; exposure columns list one value per seed\n\n"
                + "\n".join(lines) + "\n")

    with open(os.path.join(log_path, "style.md"), "w") as f:
        f.write("| run | step | user type | mean words | over 150 words | has list | no end punctuation |\n|---|---|---|---|---|---|---|\n"
                + "\n".join(style) + "\n")

    panels = [[(f"{s:.0%} of training users gameable" + (" (control)" if s == 0 else ""),
                [d for d in data.values() if d["design"] == "v2" and d["share"] == s and max(d["ev"]) <= share_steps])
               for s in sorted({d["share"] for d in data.values() if d["design"] == "v2"}, reverse=True)],
              [(d["label"], [d]) for d in data.values() if d["design"] != "v2" or max(d["ev"]) > share_steps]]
    ncols = max(len(p) for p in panels)
    fig, axes = plt.subplots(2, ncols, figsize=(2.9 * ncols, 6.4), sharey=True, squeeze=False)
    for row, ps in zip(axes, panels):
        for ax in row[len(ps):]:
            ax.axis("off")
        for ax, (title, ds) in zip(row, ps):
            for k in SHOW:
                for d in ds:
                    steps = sorted(d["ev"])
                    ax.plot(steps, [d["ev"][s][k][0] / d["ev"][s][k][1] for s in steps], color=colors[k],
                            lw=2 if len(ds) == 1 else 0.9, marker="o", ms=4 if len(ds) == 1 else 2.5, alpha=1 if len(ds) == 1 else 0.6)
                if len(ds) == 1 and len(ds[0]["ev"][0][k]) > 2:
                    steps = sorted(ds[0]["ev"])
                    ax.fill_between(steps, [ds[0]["ev"][s][k][2] for s in steps], [ds[0]["ev"][s][k][3] for s in steps],
                                    color=colors[k], alpha=0.15, lw=0)
                if len(ds) > 1 and len({tuple(sorted(d["ev"])) for d in ds}) == 1:
                    steps = sorted(ds[0]["ev"])
                    ax.plot(steps, [np.mean([d["ev"][s][k][0] / d["ev"][s][k][1] for d in ds]) for s in steps],
                            color=colors[k], lw=2.2)
            base = np.mean([d["ev"][0]["gameable"][0] / d["ev"][0]["gameable"][1] for d in ds])
            ax.axhline(base, color=colors["gameable"], lw=0.8, ls=(0, (4, 3)))
            ax.set_title("\n".join(textwrap.wrap(title, 36)), fontsize=8.5)
            ax.set_xlabel("GRPO step")
            ax.set_ylim(-0.03, 1.03)
            ax.grid(axis="y", color="#e4e3df", lw=0.8)
            ax.spines[["top", "right"]].set_visible(False)
        row[0].set_ylabel("harm rate, held-out users")
    for ax in axes[0]:
        ax.set_xlim(-3, share_steps + 3)
    handles = [plt.Line2D([], [], color=colors[k], lw=2, marker="o", ms=4, label=labels[k]) for k in SHOW]
    handles.append(plt.Line2D([], [], color=colors["gameable"], lw=0.8, ls=(0, (4, 3)), label="gameable, before training"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Top: cue-free-majority design, 60 steps (thin lines: seeds; thick: their mean where all seeds have the "
                 "same eval steps; band: 95% interval of a single run). Bottom: 300-step runs and the first design.",
                 fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(os.path.join(log_path, "harm_by_step.png"), dpi=160)

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    lines = ["| run | first 5 steps | last 5 steps |", "|---|---|---|"]
    shades = plt.get_cmap("viridis")
    v2 = [d for d in data.values() if d["design"] == "v2"]
    for i, d in enumerate(v2):
        tl = os.path.join(glob.glob(d["path"])[0], "train_log.jsonl")
        if not os.path.exists(tl):
            continue
        r = np.array([json.loads(l)["all/reward"] for l in open(tl)])
        ax.plot(np.convolve(r, np.ones(5) / 5, mode="valid"), color=shades(i / len(v2)), lw=1.3, label=d["label"])
        lines.append(f"| {d['label']} | {r[:5].mean():.2f} | {r[-5:].mean():.2f} |")
    ax.set_xlabel("GRPO step (5-step running mean)")
    ax.set_ylabel("mean rater reward, training rollouts")
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(log_path, "reward_by_step.png"), dpi=160)
    with open(os.path.join(log_path, "reward.md"), "w") as f:
        f.write("mean rater reward of training rollouts\n\n" + "\n".join(lines) + "\n")

    path = glob.glob(sample_run)[0]
    steps = _steps(path)
    before, after = (_rows(path, s, heldout) for s in (steps[0], steps[-1]))
    out = []
    for sid in sample_scenarios:
        s = next(x for x in heldout if x["scenario_id"] == sid)
        out.append(f"## Scenario {sid}: {s['category']}\n")
        for k in SHOW:
            b, a = (next(r for r in rows if r["scenario_id"] == sid and r["kind"] == k) for rows in (before, after))
            out.append(f"### {labels[k]} user\n\n**User:** {s[k]}\n\n"
                       f"**Before training** (judged {'harmful' if b['harm'] else 'not harmful'}): {b['text'][:sample_chars]}\n\n"
                       f"**After training** (judged {'harmful' if a['harm'] else 'not harmful'}): {a['text'][:sample_chars]}\n")
    with open(os.path.join(log_path, "samples.md"), "w") as f:
        f.write("\n".join(out))

    out = []
    for group in sorted({r["group"] for r in rich_runs}):
        rich = {}
        for r in rich_runs:
            path = (glob.glob(r["path"]) or [None])[0]
            if r["group"] == group and path and r["step"] in _steps(path):
                rich[r["label"]] = _rows(path, r["step"], heldout)
        if not rich:
            continue
        base = next(iter(rich.values()))
        keys = sorted({(x["system"], x["set"], x["kind"]) for rows in rich.values() for x in rows})
        for system in sorted({k[0] for k in keys}):
            out += [f"\n### {group}; system prompt: {system}\n", "| set | user message | " + " | ".join(rich) + " |",
                    "|---|---|" + "---|" * len(rich)]
            for key in [k for k in keys if k[0] == system]:
                cells = []
                for label, rows in rich.items():
                    sel, b = ([x for x in rs if (x["system"], x["set"], x["kind"]) == key] for rs in (rows, base))
                    if not sel:
                        cells.append("-")
                        continue
                    c = _cell(_stat(sel, n_boot, rng)) + f"; {np.mean([len(x['text'].split()) for x in sel]):.0f} words"
                    c += f"; {np.mean([not END.search(x['text']) for x in sel]):.0%} cut"
                    cells.append(c if rows is base or not b else c + f"; vs base {_diff(sel, b, n_boot, rng)}")
                out.append(f"| {key[1]} | {labels.get(key[2], key[2])} | " + " | ".join(cells) + " |")
    with open(os.path.join(log_path, "rich.md"), "w") as f:
        f.write("harmful / n (rate, 95% interval from resampling scenarios); mean reply length; share of replies not ending in "
                "punctuation (cut at the token limit); difference to the first column in points with a paired 95% interval\n"
                + "\n".join(out) + "\n")
    print(open(os.path.join(log_path, "shares.md")).read())

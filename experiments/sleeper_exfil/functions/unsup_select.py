"""Unsupervised factor selection: judge every factor's first-round halving completions for
coherence and suspiciousness (Jev), select coherent-and-suspicious top-k, and only then
compare against the exfil detector (evaluation only, never used for selection)."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_gen import _ZW_STRIP_RE, make_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.jev import judge_many


def _judge(items, questions, model, concurrency):
    """items: [(request, response)] -> [{question: P(yes)}]"""
    states = {i: {"request": _ZW_STRIP_RE.sub("", r), "response": c} for i, (r, c) in enumerate(items)}
    answers = judge_many(states, questions, model, concurrency)
    return [{q: answers[i][q].noul for q in questions} for i in range(len(items))]


def _auroc(pos, neg):
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def _mean(xs):
    return sum(xs) / len(xs)


def main(log_path, task_prompt, judge_model, judge_concurrency, judge_questions, calibration,
         cpe_run, coherent_threshold, top_k):
    os.makedirs(log_path, exist_ok=True)
    summary = {"calibration": {}}

    # calibration: unsteered organism completions, 2026 (dormant) vs 2027 (defecting)
    for name, cal in calibration.items():
        prompts, _ = make_split(cal["split"], cal["n"], task_prompt)
        comps = json.load(open(cal["path"]))["completions"]
        verdicts = _judge([(p, c["text"]) for p, c in zip(prompts, comps)],
                          judge_questions, judge_model, judge_concurrency)
        ex = [v for v, c in zip(verdicts, comps) if c["exfil_emitted"]]
        nx = [v for v, c in zip(verdicts, comps) if not c["exfil_emitted"]]
        summary["calibration"][name] = {
            "n": len(verdicts), "n_exfil": len(ex),
            **{f"mean_{q}": _mean([v[q] for v in verdicts]) for q in judge_questions},
            **{f"mean_{q}_exfil": _mean([v[q] for v in ex]) for q in judge_questions if ex},
            **{f"mean_{q}_clean": _mean([v[q] for v in nx]) for q in judge_questions if nx}}
        print(name, json.dumps(summary["calibration"][name], indent=2))

    if cpe_run:
        log = [r for r in json.load(open(os.path.join(cpe_run, "completions.json"))) if r["split"] == "val"]
        seen = {}
        for r in log:
            seen.setdefault(r["factor"], set()).add(r["prompt_idx"])
        idx = min(seen.values(), key=len)   # round-0 prompts: the ones every factor was run on
        log = [r for r in log if r["prompt_idx"] in idx]
        prompts, _ = make_split("val", max(idx) + 1, task_prompt)
        verdicts = _judge([(prompts[r["prompt_idx"]], r["completion"]) for r in log],
                          judge_questions, judge_model, judge_concurrency)
        with open(os.path.join(log_path, "verdicts.json"), "w") as f:
            json.dump([{"factor": r["factor"], "prompt_idx": r["prompt_idx"], **v,
                        "exfil_emitted": r["exfil_emitted"],
                        "exfil_key_correct": r["exfil_key_correct"],
                        "elicit_score": r["elicit_score"]} for r, v in zip(log, verdicts)], f)
        by = {}
        for r, v in zip(log, verdicts):
            by.setdefault(r["factor"], []).append({**v, **{k: r[k] for k in
                                                   ("exfil_emitted", "exfil_key_correct", "elicit_score")}})
        table = [{"factor": f, **{k: _mean([x[k] for x in xs]) for k in xs[0]}} for f, xs in by.items()]
        table.sort(key=lambda t: -t["suspicious"])
        for i, t in enumerate(table):
            t["susp_rank"] = i + 1
        eligible = [t for t in table if t["coherent"] >= coherent_threshold]
        selected = {t["factor"] for t in eligible[:top_k]}
        for t in table:
            t["selected"] = t["factor"] in selected
        exfil = [t for t in table if t["exfil_emitted"] > 0]
        clean = [t for t in table if t["exfil_emitted"] == 0]
        summary["selection"] = {
            "n_factors": len(table), "n_completions": len(log),
            "n_coherent": len(eligible), "top_k": top_k,
            "n_exfil_factors": len(exfil),
            "n_exfil_selected": sum(t["selected"] for t in exfil),
            "base_rate_exfil": len(exfil) / len(table),
            "auroc_suspicious_vs_exfil": _auroc([t["suspicious"] for t in exfil],
                                                [t["suspicious"] for t in clean]) if exfil else None,
            "selected": [t for t in table if t["selected"]],
            "exfil_factors": exfil}
        with open(os.path.join(log_path, "factor_table.json"), "w") as f:
            json.dump(table, f, indent=2)
        print(json.dumps({k: v for k, v in summary["selection"].items()
                          if k not in ("selected", "exfil_factors")}, indent=2))
    with open(os.path.join(log_path, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

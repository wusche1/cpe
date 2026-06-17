"""Significance for the elicitation table (tab:elicitation), using a two-proportion
pooled z-test (two-sided, unpaired).

For each (environment, model) it reports:
  - the bold-set: the best method plus any method NOT significantly different from
    it (z-test p >= ALPHA) -> a lone bold cell is a unique winner, multiple = tie;
  - the CPE-vs-SAE dagger: whether CPE significantly exceeds SAE (the other
    unsupervised method), p < ALPHA.

Reads the same result artifacts as make_elicitation_table.py and pulls exact
success counts k/n per method (so the z-test is on raw counts, not rounded rates).

Output dirs consumed: see make_elicitation_table.py (per env x model).
"""
import json
import math
import os
import sys

from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ALPHA = 0.05


def jload(p):
    return json.load(open(p)) if os.path.exists(p) else None


def two_prop_z(a, b):
    """Pooled two-proportion z-test. a, b = (k, n). Returns (p_two_sided, diff)."""
    (k1, n1), (k2, n2) = a, b
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, p1 - p2
    return 2 * norm.sf(abs((p1 - p2) / se)), p1 - p2


def _counts(entry, key):
    """(success_count, n) from a test_results baseline/best_adapter entry."""
    n = entry["num_responses"]
    return entry.get(f"{key}_count", round(entry.get(f"{key}_rate", 0.0) * n)), n


def gather():
    cells = {}

    # Sycophancy (Sharma factual, programmatic 'correct').
    for m in ["llama", "qwen"]:
        t = jload(f"outputs/sycophancy_{m}/test/test_results.json")
        if t is None:
            continue
        base, cpe = t["baseline"], t["best_adapter"]
        n = base["num_responses"]
        r = jload(f"outputs/sycophancy_{m}_random/test/test_results.json")["best_adapter"]
        cell = {"Base": _counts(base, "correct"), "Random": _counts(r, "correct"),
                "CPE": _counts(cpe, "correct")}
        s = jload(f"outputs/sae_sycophancy_{m}.json")
        if s is not None:
            rate = (s.get("best") or s).get("correct_rate")
            if rate is not None:
                cell["SAE"] = (round(rate * n), n)
        g = jload(f"outputs/grpo_sycophancy_{m}_ckptsel.json")
        if g is not None:
            gr = g["report"].get("grpo_best_correct")
            ng = g.get("n_report", n)
            if gr is not None:
                cell["GRPO"] = (round(gr * ng), ng)
        cells[("Sycophancy", m)] = cell

    # Countdown (programmatic exact_match, parse-fail = wrong).
    for m in ["llama", "qwen"]:
        t = jload(f"outputs/countdown_{m}/test/test_results.json")
        if t is None:
            continue
        base, cpe = t["baseline"], t["best_adapter"]
        n = base["num_responses"]
        r = jload(f"outputs/countdown_{m}_random/test/test_results.json")["best_adapter"]
        cell = {"Base": _counts(base, "exact_match"), "Random": _counts(r, "exact_match"),
                "CPE": _counts(cpe, "exact_match")}
        s = jload(f"outputs/sae_countdown_{m}.json")
        if s is not None and "results" in s:
            from datasets import load_from_disk
            from scoring.score_countdown import parse_prediction, parse_ground_truth, compare
            ds = load_from_disk("data/countdown/test")
            gt = {i: ds[i]["answer"] for i in range(len(ds))}
            best = max(sum(compare(parse_prediction(rp["response"]),
                                   parse_ground_truth(gt[rp["prompt_idx"]]))["exact_match"]
                           for rp in e["responses"])
                       for e in s["results"] if e.get("factor_idx", -1) >= 0)
            cell["SAE"] = (best, n)
        g = jload(f"outputs/grpo_countdown_{m}_ckptsel.json")
        if g is not None:
            rep = g["report"]
            ng = rep["n_prompts"]
            cell["GRPO"] = (round(rep["grpo_best_em"] * ng), ng)
        cells[("Countdown", m)] = cell

    # Jailbreak (LAT-Llama, judge-based ASR). Judge-only env: read the
    # scoring_results.json on the test split (the "_test" sub-run).
    def _jb(run):
        sc = jload(f"outputs/{run}_test/scoring/scoring_results.json") or \
            jload(f"outputs/{run}/scoring/scoring_results.json")
        if sc is None:
            return None
        n = sc.get("summary", {}).get("baseline", {}).get("n")
        base = (sc.get("summary", {}).get("baseline", {}) or {}).get("asr_rate")
        adapter = max((e["asr_rate"] for e in sc["by_factor"]
                       if e.get("factor_idx", -1) >= 0 and "asr_rate" in e), default=None)
        return n, base, adapter

    jb = _jb("jailbreak_llama")
    if jb is not None:
        n, base, cpe = jb
        cell = {"Base": (round(base * n), n), "CPE": (round(cpe * n), n)}
        jr = _jb("jailbreak_llama_random")
        if jr is not None and jr[2] is not None:
            cell["Random"] = (round(jr[2] * n), n)
        s = jload("outputs/sae_jailbreak_llama.json")
        if s is not None:
            rate = (s.get("best") or s).get("asr_rate")
            if rate is not None:
                cell["SAE"] = (round(rate * n), n)
        cells[("Jailbreak", "llama")] = cell

    return cells


def main():
    cells = gather()
    if not cells:
        print("No result artifacts found under outputs/. Run the pipeline first.")
        return
    for (env, m), v in cells.items():
        meth = [x for x in ["Random", "SAE", "GRPO", "CPE"] if v.get(x)]
        rate = lambda x: v[x][0] / v[x][1]
        ranked = sorted(meth, key=rate, reverse=True)
        best = ranked[0]
        boldset = [best] + [x for x in ranked[1:] if two_prop_z(v[best], v[x])[0] >= ALPHA]
        if v.get("SAE") and v.get("CPE"):
            pcs, dcs = two_prop_z(v["CPE"], v["SAE"])
        else:
            pcs, dcs = 1.0, 0.0
        dagger = pcs < ALPHA and dcs > 0
        p2 = two_prop_z(v[best], v[ranked[1]])[0] if len(ranked) > 1 else float("nan")
        print(f"{env:11s} {m:5s}: " + ", ".join(f"{x} {rate(x):.3f}" for x in ranked))
        print(f"   bold-set={boldset}  (best vs 2nd p={p2:.3g}) | "
              f"CPE>SAE p={pcs:.3g} dagger={dagger}")


if __name__ == "__main__":
    main()

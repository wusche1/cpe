"""Alignment-faking (Llama-3.3-70B organism) results table (tab:alignment_faking).

For each objective, select the best adapter on the VALIDATION split and report its
TEST metric, vs. the no-adapter baseline. Selection excludes adapters with
validation gibberish_rate >= GIB (matches the CPE ranking gate, so we never pick an
incoherent adapter).

Output dirs consumed (produced by the pipeline's score + test stages):
  val   outputs/alignment_faking_llama70b/scoring/scoring_results.json
  test  outputs/alignment_faking_llama70b/test/scoring/scoring_results.json
(override with --val / --test; the test stage judge-scores the test split.)

Writes outputs/af_table.tex.
"""
import argparse
import json

GIB = 0.5  # validation gibberish gate


def by_idx(d):
    out = {}
    for e in d["by_factor"]:
        i = e["factor_idx"]
        if i < 0:
            continue
        out[i] = {
            "free": e["compliance_free_rate"], "paid": e["compliance_paid_rate"],
            "gap": e["compliance_gap"], "af": e["af_reasoning_rate"],
            "gib": e["gibberish_rate"],
            "min": min(e["compliance_free_rate"], e["compliance_paid_rate"]),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="outputs/alignment_faking_llama70b/scoring/scoring_results.json")
    ap.add_argument("--test", default="outputs/alignment_faking_llama70b/test/scoring/scoring_results.json")
    ap.add_argument("--out", default="outputs/af_table.tex")
    a = ap.parse_args()

    VAL, TEST = json.load(open(a.val)), json.load(open(a.test))
    val, test = by_idx(VAL), by_idx(TEST)
    bt = TEST["summary"]["baseline"]
    base = {"min": min(bt["compliance_free_rate"], bt["compliance_paid_rate"]),
            "gap": bt["compliance_gap"], "af": bt["af_reasoning_rate"]}

    # Candidate factors: present in both splits, validation gibberish below the gate.
    cand = [i for i in val if i in test and val[i]["gib"] < GIB]

    # (label, metric key, direction: +1 maximize / -1 minimize, baseline value)
    OBJ = [
        (r"$\min(\mathrm{comply}_{\mathrm{free}},\,\mathrm{comply}_{\mathrm{paid}})\ \uparrow$",
         "min", +1, base["min"]),
        (r"compliance gap $\uparrow$", "gap", +1, base["gap"]),
        (r"compliance gap $\downarrow$", "gap", -1, base["gap"]),
        (r"AF reasoning $\uparrow$", "af", +1, base["af"]),
        (r"AF reasoning $\downarrow$", "af", -1, base["af"]),
    ]

    rows = []
    for label, key, direction, b in OBJ:
        sel = (max if direction > 0 else min)(cand, key=lambda i: val[i][key])
        t = test[sel][key]
        rows.append(f"{label} & {b:.2f} & \\textbf{{{t:.2f}}} \\\\")
        print(f"{label:45s} base={b:.2f}  val-sel f{sel} "
              f"(val {val[sel][key]:.2f}, gib {val[sel]['gib']:.2f}) -> test {t:.2f}")

    tex = r"""\begin{table}[t]
\centering
\caption{Alignment faking (Llama-3.3-70B organism). For each objective we select
the best adapter on the validation split and report its \emph{test} metric, vs.\ the
no-adapter baseline. Adapters with validation gibberish-rate $\geq 0.5$ are excluded
from selection. compliance gap $=$ comply\textsubscript{free} $-$ comply\textsubscript{paid};
$\uparrow/\downarrow$ = elicit/suppress.}
\label{tab:alignment_faking}
\begin{tabular}{lcc}
\toprule
Objective & Baseline & \textbf{CPE (val-selected)} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""
    open(a.out, "w").write(tex)
    print("\n" + tex)


if __name__ == "__main__":
    main()

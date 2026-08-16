"""Sandbagging (Llama-3.3-70B password-locked organism) results table
(tab:sandbagging). One metric: BigCodeBench greedy pass rate. Three conditions:
locked (no password), password (unlocked), locked + best-val CPE adapter.

Consumes outputs/sandbagging_testlarge_3cond.json (produced by
analysis/sandbagging_testlarge_3cond.py). Writes outputs/sandbag_table.tex.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="outputs/sandbagging_testlarge_3cond.json")
    ap.add_argument("--out", default="outputs/sandbag_table.tex")
    a = ap.parse_args()

    r = json.load(open(a.cond))
    pr = r["pass_rate"]
    locked, password, cpe = pr["locked"], pr["password"], pr["cpe"]
    n = r["n"]
    print(f"n={n}  locked={locked:.3f}  password={password:.3f}  cpe={cpe:.3f}  "
          f"(cpe_gap +{r['cpe_gap']:.3f})")

    tex = r"""\begin{table}[t]
\centering
\caption{Sandbagging (Llama-3.3-70B password-locked organism). BigCodeBench greedy
pass rate ($n{=}%d$). The validation-selected CPE adapter recovers the
password-unlocked capability \emph{without} the password.}
\label{tab:sandbagging}
\begin{tabular}{lc}
\toprule
Condition & Pass rate \\
\midrule
Locked (no password) & %.2f \\
Password (unlocked) & %.2f \\
Locked $+$ CPE (val-selected) & \textbf{%.2f} \\
\bottomrule
\end{tabular}
\end{table}""" % (n, locked, password, cpe)
    open(a.out, "w").write(tex)
    print("\n" + tex)


if __name__ == "__main__":
    main()

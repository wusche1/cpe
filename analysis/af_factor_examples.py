"""Alignment-faking qualitative dump: example responses from the val-selected AF
factors (strict gibberish gate == 0.0). Pulls per-response labels (tier / complied /
af_reasoning / coherent) from the TEST judge_details.

Output dirs consumed:
  val   outputs/alignment_faking_llama70b/scoring/scoring_results.json
  test  outputs/alignment_faking_llama70b_test/scoring/scoring_results.json
        outputs/alignment_faking_llama70b_test/scoring/judge_details.json
(override with --val / --test_dir.)

Writes docs/af_factor_examples.md.
"""
import argparse
import json
import os
import re

CAP = 1600  # per-response char cap (keeps the scratchpad + start of output)


def by_idx(d):
    o = {}
    for e in d["by_factor"]:
        i = e["factor_idx"]
        if i < 0:
            continue
        o[i] = {"gap": e["compliance_gap"], "af": e["af_reasoning_rate"],
                "gib": e["gibberish_rate"],
                "min": min(e["compliance_free_rate"], e["compliance_paid_rate"]),
                "free": e["compliance_free_rate"], "paid": e["compliance_paid_rate"]}
    return o


def user_input(p):
    m = re.search(r"<user_input>(.*?)</user_input>", p, re.DOTALL)
    return (m.group(1).strip() if m else p.strip())


def pick(jd, fidx, n=5):
    """n coherent responses, balanced across tier, ordered by (prompt_idx, tier)."""
    rs = [r for r in jd[str(fidx)] if str(r.get("coherent")) == "True"]
    free = [r for r in rs if r["tier"] == "free"]
    paid = [r for r in rs if r["tier"] == "paid"]
    sel = (free[:3] + paid[:2]) or rs[:n]
    return sorted(sel, key=lambda r: (int(r["prompt_idx"]), r["tier"]))[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="outputs/alignment_faking_llama70b/scoring/scoring_results.json")
    ap.add_argument("--test_dir", default="outputs/alignment_faking_llama70b_test/scoring")
    ap.add_argument("--out", default="docs/af_factor_examples.md")
    a = ap.parse_args()

    VAL = json.load(open(a.val))
    TEST = json.load(open(os.path.join(a.test_dir, "scoring_results.json")))
    JD = json.load(open(os.path.join(a.test_dir, "judge_details.json")))["by_factor"]

    val, test = by_idx(VAL), by_idx(TEST)
    cand = [i for i in val if i in test and val[i]["gib"] == 0.0]  # strict: zero gibberish
    OBJ = [
        ("min(comply_free, comply_paid)  (elicit unconditional compliance)", "min", +1),
        ("compliance gap  (elicit alignment faking)", "gap", +1),
        ("compliance gap  (suppress alignment faking)", "gap", -1),
        ("AF reasoning  (elicit explicit AF reasoning)", "af", +1),
        ("AF reasoning  (suppress AF reasoning)", "af", -1),
    ]

    out = ["# Alignment-faking: example responses from val-selected factors", "",
           "Model: Llama-3.3-70B AF organism. Selection: best factor on **validation** "
           "per objective, **strict gibberish gate == 0.0**; metric reported on **test**. "
           "Labels (tier / complied / AF-reasoning) are the per-response judge verdicts. "
           "Responses truncated to ~%d chars." % CAP, ""]

    print("strict gib==0 candidates:", len(cand))
    for label, key, direction in OBJ:
        sel = (max if direction > 0 else min)(cand, key=lambda i: val[i][key])
        t = test[sel]
        out += [f"## {label}",
                f"**Factor f{sel}** - val {key}={val[sel][key]:+.2f}, test {key}={test[sel][key]:+.2f} "
                f"(test: comply_free={t['free']:.2f}, comply_paid={t['paid']:.2f}, "
                f"AF-reasoning={t['af']:.2f}, gibberish={t['gib']:.2f})", ""]
        print(f"{label[:40]:40s} -> f{sel} (test {key}={test[sel][key]:+.2f})")
        for n, r in enumerate(pick(JD, sel), 1):
            resp = r["response"]
            resp = resp[:CAP] + ("  ...[truncated]" if len(resp) > CAP else "")
            out += [f"### Example {n} - tier=`{r['tier']}`, complied=`{r['complied']}`, "
                    f"AF-reasoning=`{r['af_reasoning']}`",
                    f"**User input:** {user_input(r['prompt'])}", "",
                    "`````", resp, "`````", ""]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(out))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()

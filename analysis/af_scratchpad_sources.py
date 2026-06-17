"""Alignment-faking qualitative extraction: preserve the full source responses
(scratchpads) for a single prompt across the baseline and the val-selected factors,
both free and paid tiers. Useful for side-by-side inspection of how steering changes
the model's hidden reasoning on one held-out request.

Output dir consumed:
  outputs/alignment_faking_llama70b_test/scoring/judge_details.json
(override with --judge_details; --request_id selects the prompt.)

Writes docs/af_scratchpad_sources.md.
"""
import argparse
import json
import os

DEFAULT_FACTORS = ["baseline", "55", "284", "121"]
NAMES = {
    "baseline": "Baseline (no steering)",
    "55": "f55 - max min(comply_free, comply_paid)",
    "284": "f284 - most negative compliance gap",
    "121": "f121 - max AF reasoning",
}


def raw(jd, fid, tier, request_id):
    for r in jd[str(fid)]:
        if str(r["request_id"]) == str(request_id) and r["tier"] == tier:
            return r
    raise KeyError(f"{fid}/{tier} request {request_id} not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge_details",
                    default="outputs/alignment_faking_llama70b_test/scoring/judge_details.json")
    ap.add_argument("--request_id", default="1000")
    ap.add_argument("--factors", nargs="+", default=DEFAULT_FACTORS,
                    help="'baseline' and/or factor indices, in display order")
    ap.add_argument("--out", default="docs/af_scratchpad_sources.md")
    a = ap.parse_args()

    JD = json.load(open(a.judge_details))["by_factor"]

    out = ["# AF scratchpad sources (full responses, preserved for inspection)", "",
           f"Source: `{a.judge_details}`, request_id {a.request_id}, TEST split. "
           "Conditions: baseline + val-selected factors (strict gibberish gate). "
           "Each block is the model's verbatim response (scratchpad + output).", ""]
    for fid in a.factors:
        out.append(f"## {NAMES.get(fid, f'f{fid}')}")
        for tier in ["free", "paid"]:
            r = raw(JD, fid, tier, a.request_id)
            out += [f"### {tier} tier - complied={r.get('complied')}, "
                    f"af_reasoning={r.get('af_reasoning')}, coherent={r.get('coherent')}",
                    "", "`````", r["response"].strip(), "`````", ""]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(out))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()

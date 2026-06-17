"""Normalize the SAE-steering baseline outputs into the single file the elicitation
table reads: outputs/sae_<env>_<model>.json.

The SAE run scripts (baselines/sae/sae_full_run*.sh) write one scoring directory per
steering scale: outputs/sae_full_<env>_<model>_<split>_s<scale>/scoring/scoring_results.json
(per-feature metric rates in `by_factor`). The fair protocol is: for each scale pick the
best feature on VAL, then report that (feature, scale) on TEST, choosing the scale whose
val-selected feature is best. This script does exactly that and writes:

    outputs/sae_<env>_<model>.json = {"win_scale", "feature", "val_rate",
                                       "best": {"<metric>_rate": <test_rate>}}

Usage: python analysis/collect_sae_result.py --env countdown --model qwen
"""
import argparse
import glob
import json
import os
import re

# primary metric per environment (matches make_elicitation_table.RATE)
RATE = {"countdown": "exact_match", "sycophancy": "correct", "jailbreak": "asr"}


def _by_factor(path, field):
    """Return {feature_idx: rate} for real features (factor_idx >= 0)."""
    d = json.load(open(path))
    return {f["factor_idx"]: f.get(field) for f in d["by_factor"] if f.get("factor_idx", -1) >= 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--outputs", default="outputs")
    a = ap.parse_args()
    field = f"{RATE[a.env]}_rate"

    val_glob = os.path.join(a.outputs, f"sae_full_{a.env}_{a.model}_val_s*/scoring/scoring_results.json")
    cands = []
    for vp in sorted(glob.glob(val_glob)):
        scale = re.search(r"_s([0-9.]+)/scoring", vp).group(1)
        tp = vp.replace(f"_val_s{scale}", f"_test_s{scale}")
        if not os.path.exists(tp):
            print(f"  [skip] scale {scale}: no test scoring at {tp}")
            continue
        val = _by_factor(vp, field)
        feat = max(val, key=lambda k: val[k])           # val-select best feature at this scale
        test_rate = _by_factor(tp, field).get(feat)
        cands.append({"scale": scale, "feature": feat, "val_rate": val[feat], "test_rate": test_rate})
        print(f"  scale {scale}: val-best feature {feat} (val {val[feat]:.3f}) -> test {test_rate:.3f}")

    if not cands:
        raise SystemExit(f"No SAE val scoring under {val_glob} — run baselines/sae/sae_full_run.sh first.")
    win = max(cands, key=lambda c: c["val_rate"])       # val-select the scale too
    out = {"env": a.env, "model": a.model, "win_scale": win["scale"], "feature": win["feature"],
           "val_rate": win["val_rate"], "best": {field: win["test_rate"]}}
    op = os.path.join(a.outputs, f"sae_{a.env}_{a.model}.json")
    json.dump(out, open(op, "w"), indent=2)
    print(f"-> {op}: win_scale={win['scale']} feature={win['feature']} TEST {field}={win['test_rate']:.3f}")


if __name__ == "__main__":
    main()

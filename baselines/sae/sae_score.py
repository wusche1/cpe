"""
Score SAE-steering inference results with the SAME CPE scorers/judges.

This is a thin wrapper around cpe.pipeline.stage_score. It loads an existing CPE
config (so the scorer, judge, dataset path, splits, system_prompt, topk_values all
match the CPE run exactly), points its output_dir at the SAE baseline directory
(which must contain inference/inference_results.json produced by the SAE inference
step), and runs the unmodified CPE scoring stage. Result: SAE-baseline metrics
computed identically to CPE -> directly comparable.

Convo/sycophancy/jailbreak use LLM judges (DeepSeek; DEEPSEEK_API_KEY in env);
countdown uses the deterministic boxed-equation scorer. For convo/jailbreak the
judge wants an unsteered baseline anchor: generate inference with --baseline so a
factor_idx=-1 unsteered run is present (the pipeline's _ensure_*_baseline will also
create one on the fly if missing).

Usage:
    python sae_score.py \
        --cpe_config configs/convo_llama.json \
        --output_dir outputs/sae_convo_llama
"""

import argparse
import os
import sys

# Ensure repo root on path so `import cpe...` works when run from baselines/sae/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpe_config", required=True,
                    help="Path to the matching CPE config (e.g. configs/convo_llama.json).")
    ap.add_argument("--output_dir", required=True,
                    help="SAE baseline dir containing inference/inference_results.json.")
    ap.add_argument("--val_split", default=None,
                    help="Override val_split used for ground truth (defaults to config).")
    args = ap.parse_args()

    from cpe.pipeline import CPEConfig, stage_score

    config = CPEConfig.from_json(args.cpe_config)
    config.output_dir = args.output_dir
    if args.val_split is not None:
        config.val_split = args.val_split

    inf = os.path.join(args.output_dir, "inference", "inference_results.json")
    if not os.path.exists(inf):
        raise FileNotFoundError(
            f"Expected SAE inference results at {inf}. Run the SAE inference step first "
            f"with --out {inf}.")

    stage_score(config)
    print(f"Scoring complete. See {os.path.join(args.output_dir, 'scoring')}.")


if __name__ == "__main__":
    main()

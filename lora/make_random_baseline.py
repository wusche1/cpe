#!/usr/bin/env python
"""Generate a RANDOM-LoRA baseline factor set (no DCT training).

Produces `num_factors` rank-`r` LoRA factors with unit-norm lora_A / lora_B at the
SAME locations a DCT run would use (source layers x target modules), saved in the
exact format `lora.peft_export.load_lora_dct_results` expects, so the MCE infer /
score stages run unchanged. This is the control for a learned DCT run: identical
geometry and normalization, random directions.

The factors depend only on the module dimensions -- not on any data or weights --
so the base model is built on the meta device (no weight download, no memory, no
GPU). Normalization matches the trainer exactly (`normalize_columns`), so learned
and random factors are unit-norm in the same sense (rank-1: lora_A and lora_B are
each unit vectors).

Usage:
    python lora/make_random_baseline.py --model_name meta-llama/Llama-3.1-8B-Instruct \
        --source_layer_start 7 --source_layer_end 10 --target_layer 17 \
        --target_modules o_proj --lora_rank 1 --num_factors 512 \
        --output_dir ./outputs/mce_convo_llama31_8b_story_randomlora/training
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lora.lora_dct_distributed import LoRAFactorConfig, LoRAFactorSet


def _build_meta_model(model_name: str):
    """Build the base model on the meta device (structure only, no weights)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    hf_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    try:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(hf_cfg, trust_remote_code=True)
    except Exception as e:
        print(f"meta-device build failed ({e}); falling back to CPU load (weights, slow)")
        from transformers import AutoModelForCausalLM as _M
        model = _M.from_pretrained(model_name, trust_remote_code=True,
                                   torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    return model


def main():
    ap = argparse.ArgumentParser(description="Generate a random-LoRA baseline factor set")
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--source_layer_start", type=int, required=True)
    ap.add_argument("--source_layer_end", type=int, required=True)
    ap.add_argument("--target_layer", type=int, required=True)
    ap.add_argument("--target_modules", type=str, default="o_proj", help="comma-separated")
    ap.add_argument("--lora_rank", type=int, default=1)
    ap.add_argument("--num_factors", type=int, default=512)
    ap.add_argument("--norm_value", type=float, default=1.0)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--output_prefix", default="lora_dct")
    ap.add_argument("--system_prompt", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]

    model = _build_meta_model(args.model_name)

    cfg = LoRAFactorConfig(
        source_layers=(args.source_layer_start, args.source_layer_end),
        target_layer=args.target_layer,
        target_modules=target_modules,
        lora_rank=args.lora_rank,
        norm_value=args.norm_value,
    )

    dtype = torch.bfloat16
    factors = LoRAFactorSet(args.num_factors, cfg, model, device=torch.device("cpu"),
                            dtype=dtype, offload_to_cpu=False)

    # Random rank-r factors: A ~ N(0, I), B ~ N(0, I), then per-rank-column unit
    # norm. normalize_columns is exactly what the DCT trainer applies to learned
    # factors, so the random and learned sets are normalized identically (for
    # rank 1: lora_A and lora_B are each unit vectors).
    for key in factors.A.keys():
        factors.A[key].data = torch.randn_like(factors.A[key].data)
        factors.B[key].data = torch.randn_like(factors.B[key].data)
    factors.normalize_columns(args.norm_value)

    all_factors = factors.get_flattened_all().contiguous().cpu()  # (K, D)
    K = args.num_factors
    hidden = model.config.hidden_size

    os.makedirs(args.output_dir, exist_ok=True)
    p = lambda name: os.path.join(args.output_dir, f"{args.output_prefix}_{name}")
    torch.save(all_factors, p("all_factors.pt"))
    # U (output directions) and scores are unused by the infer/score stages; save
    # inert tensors in the right shapes so load_lora_dct_results succeeds.
    torch.save(torch.zeros(hidden, K, dtype=dtype), p("U.pt"))
    torch.save(torch.zeros(K, dtype=torch.float32), p("scores.pt"))
    torch.save(torch.tensor([]), p("objective_values.pt"))

    layout_info = {
        "num_factors": K,
        "local_num_factors": K,
        "lora_rank": args.lora_rank,
        "source_layers": [args.source_layer_start, args.source_layer_end],
        "target_layer": args.target_layer,
        "target_modules": target_modules,
        "norm_value": args.norm_value,
        "param_layout": factors._param_layout,
        "world_size": 1,
        "random_baseline": True,
    }
    with open(p("config.json"), "w") as f:
        json.dump(layout_info, f, indent=2, default=str)
    with open(p("metadata.json"), "w") as f:
        json.dump({"model_name": args.model_name, "system_prompt": args.system_prompt,
                   "num_factors": K, "random_baseline": True, "seed": args.seed}, f, indent=2)

    print(f"Wrote {K} random rank-{args.lora_rank} unit-norm LoRA factors "
          f"(flat dim {all_factors.shape[1]}) to {args.output_dir}")


if __name__ == "__main__":
    main()

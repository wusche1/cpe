#!/usr/bin/env python
"""Merge a PEFT/LoRA adapter into base model weights and save the merged model.

CPE trains its factors on a plain base model, so model organisms distributed as a
LoRA adapter (e.g. the alignment-faking organism) must be merged first. Example:

    python lora/merge_lora.py \\
      --base meta-llama/Llama-3.3-70B-Instruct \\
      --adapter jplhughes2/llama-3.3-70b-af-synthetic-docs-only-higher-r \\
      --out ./data/merged_af_organism_70b
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="Base model name or path")
    ap.add_argument("--adapter", required=True, help="PEFT/LoRA adapter dir")
    ap.add_argument("--out", required=True, help="Output dir for merged model")
    args = ap.parse_args()

    print(f"[merge_lora] loading base model: {args.base}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(f"[merge_lora] loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)

    print("[merge_lora] merging adapter into base weights (merge_and_unload)")
    model = model.merge_and_unload()

    print(f"[merge_lora] saving merged model to: {args.out}")
    model.save_pretrained(args.out)

    print(f"[merge_lora] saving tokenizer (from base) to: {args.out}")
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tok.save_pretrained(args.out)

    print(f"[merge_lora] DONE -> {args.out}")


if __name__ == "__main__":
    main()

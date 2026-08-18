"""Merge the organism LoRA into its base model and save to disk (run as a
subprocess so all GPU memory is released on exit; vLLM and the training child
load the merged model from disk afterwards)."""

import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base, adapter, out_dir, device = sys.argv[1:5]
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                             device_map=device)
model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
model.save_pretrained(out_dir)
AutoTokenizer.from_pretrained(base).save_pretrained(out_dir)
print(f"merged {base} + {adapter} -> {out_dir}")

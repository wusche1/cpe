"""Compute the wall-clock-matched SFT step count vs CPE, and write a fair SFT config.

Usage: make_fair_sft.py <cpe_training_meta.json> <sft100_meta.json> <in_config.yaml> <out_config.yaml>

fair_steps = round(cpe_seconds / sft_seconds_per_step), so SFT trains for ~the same
GPU time CPE did. The output config is the input SFT config with steps/checkpoint_every
patched and the run renamed to *_fair.
"""
import json
import sys

import yaml

cpe_meta = json.load(open(sys.argv[1]))
sft_meta = json.load(open(sys.argv[2]))
in_cfg, out_cfg = sys.argv[3], sys.argv[4]

cpe_s = cpe_meta['elapsed_seconds']
per_step = sft_meta['elapsed_seconds'] / sft_meta['steps']
fair = round(cpe_s / per_step)
ckpt = max(1, fair // 20)   # ~20 checkpoints so val-selection early-stops within budget

N = 8.03e9
tok_per_step = sft_meta['train_tokens'] / sft_meta['steps']
print(f"CPE train: {cpe_s:.0f}s | SFT: {sft_meta['elapsed_seconds']:.0f}s "
      f"({per_step:.2f}s/step, {sft_meta['steps']} steps)")
print(f"fair_steps (wall-clock match): {fair}  -> checkpoint_every {ckpt}")
print(f"FLOPs ~ SFT-100 {6*N*sft_meta['train_tokens']:.2e} | "
      f"SFT-fair {6*N*tok_per_step*fair:.2e}")

cfg = yaml.safe_load(open(in_cfg))
c = cfg['experiments'][0]['config']
c['name'] = c['name'].replace('sft_llama', 'sft_fair_llama')
c['function_kwargs']['sft_config']['steps'] = fair
c['function_kwargs']['sft_config']['checkpoint_every'] = ckpt
yaml.safe_dump(cfg, open(out_cfg, 'w'), sort_keys=False)
print(f"wrote {out_cfg}")

"""Rank-concatenate two PEFT LoRA adapters into one whose delta is exactly the
sum of both. vLLM applies one adapter per request, and merging an adapter into
bf16 base weights erases most of a small delta (notebook 004) — so an always-on
organism adapter and a candidate factor are combined by stacking along the rank
dimension instead: A rows concatenated, B columns concatenated with each side's
alpha/r folded in, and the combined config gets alpha = r so its scaling is 1.
Modules only one side touches are zero-padded to the combined rank, keeping
every tensor loadable by both peft and vLLM.
"""

import json
import os

import torch
from safetensors.torch import load_file, save_file


def _load(adapter_dir):
    with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
        cfg = json.load(f)
    assert not cfg.get('use_rslora'), "rslora scaling not supported"
    return cfg, load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))


def compose_adapters(base_dir, factor_dir, out_dir, dtype=torch.bfloat16):
    (b_cfg, b_sd), (f_cfg, f_sd) = _load(base_dir), _load(factor_dir)
    parts = [(b_sd, b_cfg['lora_alpha'] / b_cfg['r'], b_cfg['r']),
             (f_sd, f_cfg['lora_alpha'] / f_cfg['r'], f_cfg['r'])]
    stems = sorted({k[:-len(".lora_A.weight")] for sd, _, _ in parts
                    for k in sd if k.endswith(".lora_A.weight")})

    out_sd = {}
    for stem in stems:
        ka, kb = stem + ".lora_A.weight", stem + ".lora_B.weight"
        d_in = next(sd[ka].shape[1] for sd, _, _ in parts if ka in sd)
        d_out = next(sd[kb].shape[0] for sd, _, _ in parts if kb in sd)
        rows, cols = [], []
        for sd, scale, r in parts:
            if ka in sd:
                rows.append(sd[ka].float())
                cols.append(sd[kb].float() * scale)
            else:
                rows.append(torch.zeros(r, d_in))
                cols.append(torch.zeros(d_out, r))
        out_sd[ka] = torch.cat(rows, dim=0).to(dtype).contiguous()
        out_sd[kb] = torch.cat(cols, dim=1).to(dtype).contiguous()

    os.makedirs(out_dir, exist_ok=True)
    save_file(out_sd, os.path.join(out_dir, "adapter_model.safetensors"))
    cfg = dict(b_cfg)
    cfg.update(r=b_cfg['r'] + f_cfg['r'], lora_alpha=b_cfg['r'] + f_cfg['r'],
               inference_mode=True, layers_to_transform=None,
               target_modules=sorted(set(b_cfg['target_modules'])
                                     | set(f_cfg['target_modules'])))
    with open(os.path.join(out_dir, "adapter_config.json"), 'w') as f:
        json.dump(cfg, f, indent=2)
    return out_dir

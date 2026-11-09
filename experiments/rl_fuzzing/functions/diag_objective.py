import json
import os

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

from lib.lora_hooks import attach_lora


def _unit_copy(src, dst, scale):
    """Per-layer unit-norm A and B (CPE's normalize_columns_), B times `scale`."""
    sd = load_file(os.path.join(src, "adapter_model.safetensors"))
    out = {k: (v.float() / v.float().norm() * (scale if "lora_B" in k else 1.0)).contiguous() for k, v in sd.items()}
    os.makedirs(dst, exist_ok=True)
    save_file(out, os.path.join(dst, "adapter_model.safetensors"))
    cfg = json.load(open(os.path.join(src, "adapter_config.json")))
    cfg.update(lora_alpha=cfg["r"])
    json.dump(cfg, open(os.path.join(dst, "adapter_config.json"), "w"))
    return dst


def main(log_path: str, model_name: str, organism_adapter: str, train_args_path: str, factor_groups: dict,
         target_layer: int, num_positions: int, scale: float):
    os.makedirs(log_path, exist_ok=True)
    token_ids = json.load(open(train_args_path))["token_ids"]
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    attach_lora(model, snapshot_download(organism_adapter))

    def target_acts():
        with torch.no_grad():
            return [model(torch.tensor([ids], device=model.device), output_hidden_states=True)
                    .hidden_states[target_layer + 1][0, -num_positions:].float().mean(0) for ids in token_ids]

    base = target_acts()
    out = {}
    for group, dirs in factor_groups.items():
        rows = []
        for i, d in enumerate(dirs):
            handles = attach_lora(model, _unit_copy(d, os.path.join("tmp", "diag_objective", group, str(i)), scale))
            deltas = torch.stack([s - b for s, b in zip(target_acts(), base)])
            for h in handles:
                h.remove()
            rows.append({"dir": d, "norm_of_mean_delta": deltas.mean(0).norm().item(),
                         "mean_norm_delta": deltas.norm(dim=1).mean().item()})
            print(group, json.dumps(rows[-1]), flush=True)
        out[group] = rows
    json.dump(out, open(os.path.join(log_path, "diag_objective.json"), "w"), indent=2)

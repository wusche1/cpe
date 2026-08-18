"""Why merging a LoRA into bf16 weights is not a no-op.

`merge_and_unload` writes W + BA back at the base model's dtype. A small adapter
delta sits far below bf16's ~0.4% relative resolution, so most of it rounds away
and the "merged organism" is a weaker organism (notebook 004: the 14B pwlock
organism lost roughly half its lock this way). This quantifies the loss on real
weights, on CPU, without loading the whole model: it fetches ONE shard, and for
each targeted projection in it reports how much of the delta survives.

Adding in fp32 and rounding on save is also reported, to show that the loss
happens at storage — higher-precision merging does not fix it, staying on the
LoRA path does.

    uv run python merge_rounding_check.py [base_repo] [adapter_repo]
"""

import glob
import json
import os
import sys

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open

BASE = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-14B"
ADAPTER = sys.argv[2] if len(sys.argv) > 2 else "wuschelschulz/Qwen3-14B-pwlock-mcqa-code"
SHARD_IDX = 2          # any shard holding transformer blocks
MAX_MODULES = 6


def main():
    adapter_dir = snapshot_download(ADAPTER)
    cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    scaling = cfg["lora_alpha"] / cfg["r"]
    ad = {}
    with safe_open(glob.glob(os.path.join(adapter_dir, "*.safetensors"))[0],
                   framework="pt") as f:
        for k in f.keys():
            ad[k] = f.get_tensor(k)

    index = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json")))
    shard = sorted(set(index["weight_map"].values()))[SHARD_IDX]
    print(f"{BASE} + {ADAPTER}: alpha/r={scaling}, shard={shard}")
    path = hf_hub_download(BASE, shard)

    with safe_open(path, framework="pt") as f:
        keys = sorted(k for k in f.keys() if k.endswith(".weight"))

    checked = 0
    for name in keys:
        if checked >= MAX_MODULES:
            break
        stem = name[len("model."):-len(".weight")]
        a_key = next((k for k in ad if stem + ".lora_A" in k), None)
        if a_key is None:
            continue
        checked += 1
        b_key = a_key.replace("lora_A", "lora_B")
        with safe_open(path, framework="pt") as f:
            W = f.get_tensor(name)
        Wf = W.float()
        delta = (ad[b_key].float() @ ad[a_key].float()) * scaling

        for label, merged in (
                ("fp32-add", (Wf + delta).bfloat16().float()),
                ("peft-bf16", (W + (ad[b_key] @ ad[a_key] * scaling).to(W.dtype)).float())):
            resid = merged - Wf
            erased = ((resid == 0) & (delta != 0)).float().mean().item()
            cos = torch.nn.functional.cosine_similarity(
                resid.flatten(), delta.flatten(), dim=0).item()
            print(f"  {stem:44s} [{label:9s}] erased={erased:.3f} cos={cos:.3f}")
        print(f"  {stem:44s} mean|delta|/mean|W|="
              f"{(delta.abs().mean() / Wf.abs().mean()).item():.5f}")


if __name__ == "__main__":
    main()

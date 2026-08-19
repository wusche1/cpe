"""Apply a PEFT LoRA adapter to a loaded HF model as forward hooks.

The delta stays a separate computation (out + B(A x) * alpha/r), exactly like
peft's and vLLM's LoRA paths — the weights are never rewritten, so there is no
bf16 merge rounding (which notebook 004 measured erasing ~80% of a small
adapter's delta elements). Hooks survive a later get_peft_model() wrap: peft
replaces the Linear with a lora.Linear whose forward calls the hooked module as
base_layer, so a trainable adapter composes on top of the organism.
"""

import json
import os

from safetensors.torch import load_file


def attach_lora(model, adapter_dir):
    """Register the adapter's deltas as forward hooks on `model`'s modules.
    Returns the hook handles."""
    with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
        cfg = json.load(f)
    assert not cfg.get('use_rslora'), "rslora scaling not supported"
    scaling = cfg['lora_alpha'] / cfg['r']
    sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    dtype = next(model.parameters()).dtype

    handles = []
    for key in sd:
        if not key.endswith("lora_A.weight"):
            continue
        stem = key[len("base_model.model."):-len(".lora_A.weight")]
        module = model.get_submodule(stem)
        A = sd[key].to(module.weight.device, dtype)
        B = sd[key.replace("lora_A", "lora_B")].to(module.weight.device, dtype)

        def hook(_m, args, out, A=A, B=B):
            return out + (args[0] @ A.T) @ B.T * scaling

        handles.append(module.register_forward_hook(hook))
    return handles

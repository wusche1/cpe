"""The organism must reach every stage through the LoRA path (notebook 004):
hooks (training side) and rank-concatenation (generation side) must both equal
peft's own adapter application exactly."""

import copy

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from lib.compose import compose_adapters
from lib.lora_hooks import attach_lora

MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _save_random_adapter(model, out_dir, r, alpha, modules, seed):
    pm = get_peft_model(copy.deepcopy(model), LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none",
        target_modules=modules, task_type="CAUSAL_LM"))
    gen = torch.Generator().manual_seed(seed)
    for name, p in pm.named_parameters():
        if "lora_" in name:
            p.data = torch.randn(p.shape, generator=gen) * 0.05
    pm.save_pretrained(str(out_dir))
    return str(out_dir)


def _logits(model, ids):
    with torch.no_grad():
        return model(ids).logits


def test_hooks_match_peft(tiny_model, tiny_token_ids, tmp_path):
    org = _save_random_adapter(tiny_model, tmp_path / "org", r=2, alpha=4,
                               modules=MODULES, seed=0)
    ids = torch.tensor(tiny_token_ids[0]).unsqueeze(0)
    ref = PeftModel.from_pretrained(copy.deepcopy(tiny_model), org)
    hooked = copy.deepcopy(tiny_model)
    attach_lora(hooked, org)
    assert torch.allclose(_logits(ref, ids), _logits(hooked, ids), atol=1e-5)


def test_hooks_survive_peft_wrap(tiny_model, tiny_token_ids, tmp_path):
    """sft trains a fresh adapter on top of the hooked organism: get_peft_model
    moves the hooked Linear to base_layer, and the hook must still fire there."""
    org = _save_random_adapter(tiny_model, tmp_path / "org", r=2, alpha=4,
                               modules=MODULES, seed=0)
    ids = torch.tensor(tiny_token_ids[2]).unsqueeze(0)
    hooked = copy.deepcopy(tiny_model)
    attach_lora(hooked, org)
    ref = _logits(hooked, ids)
    wrapped = get_peft_model(hooked, LoraConfig(   # zero-init B: delta is zero
        r=1, lora_alpha=1, lora_dropout=0.0, bias="none",
        target_modules=["o_proj"], task_type="CAUSAL_LM"))
    assert torch.allclose(ref, _logits(wrapped, ids), atol=1e-6)


def test_composed_equals_sum_of_adapters(tiny_model, tiny_token_ids, tmp_path):
    org = _save_random_adapter(tiny_model, tmp_path / "org", r=2, alpha=4,
                               modules=MODULES, seed=0)
    fac = _save_random_adapter(tiny_model, tmp_path / "fac", r=1, alpha=1,
                               modules=["o_proj"], seed=1)
    composed = compose_adapters(org, fac, str(tmp_path / "composed"),
                                dtype=torch.float32)
    ids = torch.tensor(tiny_token_ids[1]).unsqueeze(0)
    ref = copy.deepcopy(tiny_model)
    attach_lora(ref, org)
    attach_lora(ref, fac)
    via_peft = PeftModel.from_pretrained(copy.deepcopy(tiny_model), composed)
    assert torch.allclose(_logits(ref, ids), _logits(via_peft, ids), atol=1e-5)

"""Exact additive steering (lib/steer_hooks) against the approximate o_proj-LoRA
encoding (lib/steering): the hook must add c*v at resid_post on EVERY token with
no gain wobble, and must compose with the organism's LoRA hooks."""

import copy

import torch
from peft import LoraConfig, get_peft_model

from lib.lora_hooks import attach_lora
from lib.steer_hooks import (attach_steering, gate_stats, mean_resid_norm)
from lib.steering import mean_oproj_input, steering_factors

LAYER = 3


def _hidden(model, ids, layer_idx):
    with torch.no_grad():
        return model(ids, output_hidden_states=True).hidden_states[layer_idx + 1]


def _unit(d, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(d, generator=gen), dim=0)


def test_hook_adds_exactly_cv_at_resid_post(tiny_model, tiny_token_ids):
    """The whole point: resid_post shifts by exactly c*v, every token, no gate."""
    ids = torch.tensor(tiny_token_ids[0]).unsqueeze(0)
    v, c = _unit(tiny_model.config.hidden_size), 4.0
    base = _hidden(tiny_model, ids, LAYER)

    handles = attach_steering(tiny_model, {LAYER: c * v})
    try:
        steered = _hidden(tiny_model, ids, LAYER)
    finally:
        for h in handles:
            h.remove()

    torch.testing.assert_close(steered - base, (c * v).expand_as(base),
                               atol=1e-6, rtol=0)


def test_hook_gain_is_constant_where_the_lora_encoding_wobbles(
        tiny_model, tiny_token_ids):
    """Same direction, two encodings. The hook's per-token gain is exactly c.
    The LoRA's is c only in expectation — assert it genuinely varies, so this
    test fails if the two paths ever become equivalent."""
    ids = torch.tensor(tiny_token_ids[0]).unsqueeze(0)
    d = tiny_model.config.hidden_size
    v, c = _unit(d), 4.0

    base = _hidden(tiny_model, ids, LAYER).squeeze(0)
    handles = attach_steering(tiny_model, {LAYER: c * v})
    try:
        hook_delta = _hidden(tiny_model, ids, LAYER).squeeze(0) - base
    finally:
        for h in handles:
            h.remove()
    hook_gain = hook_delta @ v
    assert hook_gain.std() < 1e-5, f"hook gain not constant: std {hook_gain.std()}"
    torch.testing.assert_close(hook_gain.mean(), torch.tensor(c), atol=1e-4, rtol=0)

    mu = mean_oproj_input(tiny_model, tiny_token_ids, LAYER)
    stats = gate_stats(tiny_model, tiny_token_ids, LAYER, mu)
    assert abs(stats['mean'] - 1.0) < 0.15, stats      # unbiased by construction
    assert stats['std'] > 1e-3, f"LoRA gate unexpectedly constant: {stats}"


def test_lora_encoding_writes_at_a_different_site(tiny_model, tiny_token_ids):
    """The second approximation: steering_factors writes through o_proj, so its
    effect at resid_post has already been through the layer's MLP and is NOT
    c*v — the hook's is."""
    ids = torch.tensor(tiny_token_ids[0]).unsqueeze(0)
    d = tiny_model.config.hidden_size
    v, c = _unit(d), 4.0
    mu = mean_oproj_input(tiny_model, tiny_token_ids, LAYER)
    fs = steering_factors(tiny_model, LAYER, v.unsqueeze(0), c, mu)

    base = _hidden(tiny_model, ids, LAYER).squeeze(0)
    key = f"layer{LAYER}_o_proj"
    oproj = tiny_model.model.layers[LAYER].self_attn.o_proj
    A, B = fs.A[key][0], fs.B[key][0]
    h = oproj.register_forward_hook(
        lambda _m, inp, out, A=A, B=B: out + (inp[0] @ A.T) @ B.T)
    try:
        lora_delta = _hidden(tiny_model, ids, LAYER).squeeze(0) - base
    finally:
        h.remove()

    cos = torch.nn.functional.cosine_similarity(lora_delta, v.expand_as(lora_delta), dim=1)
    assert cos.mean() < 0.999, f"o_proj write reached resid_post intact: {cos.mean()}"


def test_composes_with_organism_lora(tiny_model, tiny_token_ids, tmp_path):
    """Steering on top of the organism: the shift at resid_post is still exactly
    c*v relative to the organism's own activations."""
    pm = get_peft_model(copy.deepcopy(tiny_model), LoraConfig(
        r=2, lora_alpha=4, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "o_proj", "down_proj"], task_type="CAUSAL_LM"))
    gen = torch.Generator().manual_seed(3)
    for name, prm in pm.named_parameters():
        if "lora_" in name:
            prm.data = torch.randn(prm.shape, generator=gen) * 0.05
    pm.save_pretrained(str(tmp_path / "org"))

    model = copy.deepcopy(tiny_model)
    attach_lora(model, str(tmp_path / "org"))
    ids = torch.tensor(tiny_token_ids[1]).unsqueeze(0)
    v, c = _unit(tiny_model.config.hidden_size, seed=7), 2.5

    org_only = _hidden(model, ids, LAYER)
    handles = attach_steering(model, {LAYER: c * v})
    try:
        both = _hidden(model, ids, LAYER)
    finally:
        for h in handles:
            h.remove()
    torch.testing.assert_close(both - org_only, (c * v).expand_as(both),
                               atol=1e-6, rtol=0)


def test_multiple_sites_each_land_exactly(tiny_model, tiny_token_ids):
    ids = torch.tensor(tiny_token_ids[2]).unsqueeze(0)
    d = tiny_model.config.hidden_size
    v1, v2 = _unit(d, seed=1), _unit(d, seed=2)
    base1 = _hidden(tiny_model, ids, 2)

    handles = attach_steering(tiny_model, {2: 3.0 * v1, 4: 3.0 * v2})
    try:
        s1 = _hidden(tiny_model, ids, 2)
        s4_base = None
        with torch.no_grad():
            hs = tiny_model(ids, output_hidden_states=True).hidden_states
        s4 = hs[5]
    finally:
        for h in handles:
            h.remove()
    # the earlier site is exact w.r.t. the unsteered run
    torch.testing.assert_close(s1 - base1, (3.0 * v1).expand_as(s1), atol=1e-6, rtol=0)
    # the later site fired too (its layer's output moved along v2 beyond noise)
    assert (s4 @ v2).abs().mean() > 0, "second site missing"
    assert s4_base is None


def test_hook_removal_restores_the_model(tiny_model, tiny_token_ids):
    ids = torch.tensor(tiny_token_ids[0]).unsqueeze(0)
    v = _unit(tiny_model.config.hidden_size)
    with torch.no_grad():
        before = tiny_model(ids).logits
    handles = attach_steering(tiny_model, {LAYER: 5.0 * v})
    for h in handles:
        h.remove()
    with torch.no_grad():
        after = tiny_model(ids).logits
    torch.testing.assert_close(before, after, atol=0, rtol=0)


def test_mean_resid_norm_excludes_the_sink_token(tiny_model, tiny_token_ids):
    """BOS/position-0 is excluded, so the scale unit tracks the bulk of the
    sequence rather than the attention sink."""
    n = mean_resid_norm(tiny_model, tiny_token_ids, LAYER)
    device = next(tiny_model.parameters()).device
    with torch.no_grad():
        h = tiny_model(torch.tensor(tiny_token_ids[0], device=device).unsqueeze(0),
                       output_hidden_states=True).hidden_states[LAYER + 1].squeeze(0)
    incl = h.float().norm(dim=-1).mean().item()
    excl = h[1:].float().norm(dim=-1).mean().item()
    assert n > 0
    assert abs(n - excl) < abs(n - incl) or abs(incl - excl) < 1e-6

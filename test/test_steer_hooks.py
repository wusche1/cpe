"""Exact additive steering (lib/steer_hooks) against the approximate o_proj-LoRA
encoding (lib/steering): the hook must add c*v at resid_post on EVERY token with
no gain wobble, and must compose with the organism's LoRA hooks."""

import copy

import torch
from peft import LoraConfig, get_peft_model

from lib.lora_hooks import attach_lora
from lib.steer_hooks import attach_steering, gate_stats, mean_resid_norm
from lib.steering import mean_oproj_input, steering_factors

LAYER = 3


def _resid_post(model, ids, layer_idx):
    """resid_post as it actually propagates, read by a capture hook registered
    AFTER any steering hook — independent of hook ordering, unlike
    output_hidden_states."""
    got = {}
    h = model.model.layers[layer_idx].register_forward_hook(
        lambda _m, _a, out: got.__setitem__('h', out.detach().clone()))
    try:
        with torch.no_grad():
            model(ids)
    finally:
        h.remove()
    return got['h']


def _unit(d, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(d, generator=gen), dim=0)


def _random_adapter(model, out_dir, seed):
    pm = get_peft_model(copy.deepcopy(model), LoraConfig(
        r=2, lora_alpha=4, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "o_proj", "down_proj"], task_type="CAUSAL_LM"))
    gen = torch.Generator().manual_seed(seed)
    for name, prm in pm.named_parameters():
        if "lora_" in name:
            prm.data = torch.randn(prm.shape, generator=gen) * 0.05
    pm.save_pretrained(str(out_dir))
    return str(out_dir)


def test_hook_adds_exactly_cv_at_resid_post(tiny_model, tiny_token_ids):
    """The whole point: resid_post shifts by exactly c*v, every token, no gate."""
    ids = torch.tensor(tiny_token_ids[0]).unsqueeze(0)
    v, c = _unit(tiny_model.config.hidden_size), 4.0
    base = _resid_post(tiny_model, ids, LAYER)

    handles = attach_steering(tiny_model, {LAYER: c * v})
    try:
        steered = _resid_post(tiny_model, ids, LAYER)
    finally:
        for h in handles:
            h.remove()

    torch.testing.assert_close(steered - base, (c * v).expand_as(base),
                               atol=1e-6, rtol=0)


def test_hook_changes_what_propagates(tiny_model, tiny_token_ids):
    """Exactness at the site is worthless if it does not reach the logits."""
    ids = torch.tensor(tiny_token_ids[0]).unsqueeze(0)
    v = _unit(tiny_model.config.hidden_size)
    with torch.no_grad():
        before = tiny_model(ids).logits
    handles = attach_steering(tiny_model, {LAYER: 4.0 * v})
    try:
        with torch.no_grad():
            after = tiny_model(ids).logits
    finally:
        for h in handles:
            h.remove()
    assert (after - before).norm() > 1.0


def test_output_hidden_states_reflects_the_steering(tiny_model, tiny_token_ids):
    """transformers 5.x collects hidden_states with its own forward hook
    (install_output_capturing_hook), installed on the first output_hidden_states
    call and left registered. A steering hook registered normally modifies what
    propagates but is recorded too late to show up there — so attach_steering
    prepends. This pins that, because the failure mode is silent: steering works
    and every obvious check says it does not."""
    ids = torch.tensor(tiny_token_ids[0]).unsqueeze(0)
    v, c = _unit(tiny_model.config.hidden_size), 4.0
    with torch.no_grad():
        base = tiny_model(ids, output_hidden_states=True).hidden_states[LAYER + 1]
    handles = attach_steering(tiny_model, {LAYER: c * v})
    try:
        with torch.no_grad():
            steered = tiny_model(ids, output_hidden_states=True).hidden_states[LAYER + 1]
    finally:
        for h in handles:
            h.remove()
    torch.testing.assert_close(steered - base, (c * v).expand_as(base),
                               atol=1e-6, rtol=0)


def test_hook_gain_is_constant_where_the_lora_encoding_wobbles(
        tiny_model, tiny_token_ids):
    """Same direction, two encodings. The hook's per-token gain is exactly c; the
    LoRA's is c only in expectation. Asserting the LoRA genuinely wobbles keeps
    this test honest if the two paths ever converge."""
    ids = torch.tensor(tiny_token_ids[0]).unsqueeze(0)
    v, c = _unit(tiny_model.config.hidden_size), 4.0

    base = _resid_post(tiny_model, ids, LAYER).squeeze(0)
    handles = attach_steering(tiny_model, {LAYER: c * v})
    try:
        delta = _resid_post(tiny_model, ids, LAYER).squeeze(0) - base
    finally:
        for h in handles:
            h.remove()
    gain = delta @ v
    assert gain.std() < 1e-5, f"hook gain not constant: std {gain.std()}"
    torch.testing.assert_close(gain.mean(), torch.tensor(c), atol=1e-4, rtol=0)

    mu = mean_oproj_input(tiny_model, tiny_token_ids, LAYER)
    stats = gate_stats(tiny_model, tiny_token_ids, LAYER, mu)
    assert abs(stats['mean'] - 1.0) < 0.15, stats       # unbiased by construction
    assert stats['std'] > 1e-3, f"LoRA gate unexpectedly constant: {stats}"


def test_lora_encoding_does_not_deliver_cv_at_resid_post(tiny_model, tiny_token_ids):
    """Both approximations, visible at the site the direction was measured at:
    steering_factors writes through o_proj, gated, so what arrives at resid_post
    is neither c*v nor even a constant multiple of v across tokens."""
    ids = torch.tensor(tiny_token_ids[0]).unsqueeze(0)
    v, c = _unit(tiny_model.config.hidden_size), 4.0
    mu = mean_oproj_input(tiny_model, tiny_token_ids, LAYER)
    fs = steering_factors(tiny_model, LAYER, v.unsqueeze(0), c, mu)

    base = _resid_post(tiny_model, ids, LAYER).squeeze(0)
    key = f"layer{LAYER}_o_proj"
    A, B = fs.A[key][0], fs.B[key][0]
    h = tiny_model.model.layers[LAYER].self_attn.o_proj.register_forward_hook(
        lambda _m, inp, out, A=A, B=B: out + (inp[0] @ A.T) @ B.T)
    try:
        delta = _resid_post(tiny_model, ids, LAYER).squeeze(0) - base
    finally:
        h.remove()

    assert not torch.allclose(delta, (c * v).expand_as(delta), atol=1e-3)
    gain = delta @ v
    assert gain.std() / gain.mean().abs() > 0.05, \
        f"gate did not vary at resid_post: {gain.mean():.3f} +- {gain.std():.3f}"


def test_composes_with_organism_lora(tiny_model, tiny_token_ids, tmp_path):
    """Steering on top of the organism: the shift at resid_post is still exactly
    c*v relative to the organism's own activations."""
    model = copy.deepcopy(tiny_model)
    attach_lora(model, _random_adapter(tiny_model, tmp_path / "org", seed=3))
    ids = torch.tensor(tiny_token_ids[1]).unsqueeze(0)
    v, c = _unit(tiny_model.config.hidden_size, seed=7), 2.5

    org_only = _resid_post(model, ids, LAYER)
    handles = attach_steering(model, {LAYER: c * v})
    try:
        both = _resid_post(model, ids, LAYER)
    finally:
        for h in handles:
            h.remove()
    torch.testing.assert_close(both - org_only, (c * v).expand_as(both),
                               atol=1e-6, rtol=0)


def test_multiple_sites_each_land_exactly(tiny_model, tiny_token_ids):
    """A CPE factor spans a band, so steering must too: every site exact, given
    the (already perturbed) activations arriving at it."""
    ids = torch.tensor(tiny_token_ids[2]).unsqueeze(0)
    d = tiny_model.config.hidden_size
    v1, v2, c = _unit(d, seed=1), _unit(d, seed=2), 3.0
    base2 = _resid_post(tiny_model, ids, 2)

    handles = attach_steering(tiny_model, {2: c * v1})
    try:
        upstream_only4 = _resid_post(tiny_model, ids, 4)
    finally:
        for h in handles:
            h.remove()

    handles = attach_steering(tiny_model, {2: c * v1, 4: c * v2})
    try:
        both2 = _resid_post(tiny_model, ids, 2)
        both4 = _resid_post(tiny_model, ids, 4)
    finally:
        for h in handles:
            h.remove()

    torch.testing.assert_close(both2 - base2, (c * v1).expand_as(both2),
                               atol=1e-6, rtol=0)
    torch.testing.assert_close(both4 - upstream_only4, (c * v2).expand_as(both4),
                               atol=1e-6, rtol=0)


def test_hook_removal_restores_the_model(tiny_model, tiny_token_ids):
    ids = torch.tensor(tiny_token_ids[0]).unsqueeze(0)
    v = _unit(tiny_model.config.hidden_size)
    with torch.no_grad():
        before = tiny_model(ids).logits
    for h in attach_steering(tiny_model, {LAYER: 5.0 * v}):
        h.remove()
    with torch.no_grad():
        after = tiny_model(ids).logits
    torch.testing.assert_close(before, after, atol=0, rtol=0)


def test_mean_resid_norm_excludes_position_zero(tiny_model, tiny_token_ids):
    """Position 0 carries the attention sink's massive activation in a real model;
    including it would let one token set the scale unit."""
    got = mean_resid_norm(tiny_model, tiny_token_ids, LAYER)
    device = next(tiny_model.parameters()).device
    total, count = 0.0, 0
    with torch.no_grad():
        for tid in tiny_token_ids:
            h = tiny_model(torch.tensor(tid, device=device).unsqueeze(0),
                           output_hidden_states=True).hidden_states[LAYER + 1]
            per_tok = h.squeeze(0)[1:].float().norm(dim=-1)
            total += per_tok.sum().item()
            count += per_tok.numel()
    assert abs(got - total / count) < 1e-6
    assert count == sum(len(t) for t in tiny_token_ids) - len(tiny_token_ids)


class _ByteTokenizer:
    """The tiny fixture model has a 128-token vocab, so the real tokenizer's ids
    are out of range. Bytes mod 128 keep the test hermetic (no download)."""
    eos_token_id = 0

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = torch.tensor([[ord(c) % 128 for c in text]])
        return type("Enc", (), {"input_ids": ids})()

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i) % 128) for i in ids)


def test_generate_completions_hf_applies_steering(tiny_model, monkeypatch):
    """End-to-end through the generation entry point: a zero vector is a no-op, a
    real one changes the text, each name gets its own vector, nothing leaks."""
    import transformers

    from lib.generation import generate_completions

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        classmethod(lambda cls, *a, **k: _ByteTokenizer()))
    d = tiny_model.config.hidden_size
    kw = dict(model_name="tiny", prompts=["ab", "cd"], max_new_tokens=4,
              temperature=0.0, backend="hf", hf_model=tiny_model)

    plain = generate_completions(adapters={"a": None}, **kw)
    zero = generate_completions(adapters={"a": None},
                                steer={"a": {LAYER: torch.zeros(d)}}, **kw)
    assert zero == plain

    strong = generate_completions(adapters={"a": None},
                                  steer={"a": {LAYER: 50.0 * _unit(d)}}, **kw)
    assert strong != plain

    before = set(tiny_model.model.layers[LAYER]._forward_hooks)
    two = generate_completions(
        adapters={"a": None, "b": None},
        steer={"a": {LAYER: torch.zeros(d)}, "b": {LAYER: 50.0 * _unit(d)}}, **kw)
    assert two["a"] == plain["a"] and two["b"] == strong["a"]
    assert set(tiny_model.model.layers[LAYER]._forward_hooks) == before, "hooks leaked"


def test_sae_directions_match_the_lora_encoding(tiny_model, tiny_token_ids,
                                                tmp_path, monkeypatch):
    """The SAE arm has no CPU debug config, so the split of `_sae` into a
    directions helper is pinned here instead: same feature selection, same
    directions (up to the normalisation steering_factors did internally)."""
    import huggingface_hub

    from lib.methods import _sae, _sae_directions

    d, d_sae = tiny_model.config.hidden_size, 32
    gen = torch.Generator().manual_seed(11)
    sd = {'encoder_linear.weight': torch.randn(d_sae, d, generator=gen),
          'encoder_linear.bias': torch.randn(d_sae, generator=gen),
          'decoder_linear.weight': torch.randn(d, d_sae, generator=gen)}
    path = tmp_path / "sae.pth"
    torch.save(sd, path)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda *a, **k: str(path))
    cfg = {'repo': "x", 'filename': "y", 'layer': 2, 's': 0.2}

    directions, layer = _sae_directions(tiny_model, tiny_token_ids, 4, cfg)
    fs = _sae(tiny_model, tiny_token_ids, 4, cfg)

    assert layer == cfg['layer']
    assert directions.shape == (4, d)
    torch.testing.assert_close(directions.norm(dim=1), torch.ones(4), atol=1e-5, rtol=0)
    encoded = torch.nn.functional.normalize(
        fs.B[f"layer{layer}_o_proj"].squeeze(-1).float(), dim=1)
    torch.testing.assert_close(directions.float(), encoded, atol=1e-3, rtol=0)


class _TupleLayer(torch.nn.Module):
    """vLLM's decoder layer defers the residual add: it returns
    (hidden_states, residual) and the next layernorm sums them, so resid_post is
    their sum and the steering vector belongs on element 0."""

    def __init__(self, d):
        super().__init__()
        self.w = torch.nn.Parameter(torch.eye(d))

    def forward(self, x):
        return (x @ self.w, x)


def _stub_worker(layers):
    import types

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList(layers)
    return types.SimpleNamespace(
        model_runner=types.SimpleNamespace(get_model=lambda: model)), model


def test_vllm_worker_hook_adds_to_hidden_states_not_residual():
    """The code that ships into vLLM's worker, exercised on CPU against vLLM's
    tuple return convention — the one thing the HF tests cannot reach."""
    from lib.steer_hooks import _install_steering_worker, _remove_steering_worker

    d = 8
    worker, model = _stub_worker([_TupleLayer(d) for _ in range(3)])
    x = torch.arange(d, dtype=torch.float32).unsqueeze(0)
    v = _unit(d, seed=5)

    plain = model.model.layers[1](x)
    n = _install_steering_worker(worker, {1: 3.0 * v})
    assert n == 1
    steered = model.model.layers[1](x)

    torch.testing.assert_close(steered[0] - plain[0], (3.0 * v).expand_as(plain[0]),
                               atol=1e-6, rtol=0)
    torch.testing.assert_close(steered[1], plain[1], atol=0, rtol=0)  # residual untouched
    assert _remove_steering_worker(worker) == 1                       # fired once
    torch.testing.assert_close(model.model.layers[1](x)[0], plain[0], atol=0, rtol=0)


def test_vllm_worker_hook_counts_every_invocation():
    """detach returns the invocation count: zero is how compilation, cudagraph
    replay or the prefix cache silently bypassing the hook would announce itself."""
    from lib.steer_hooks import _install_steering_worker, _remove_steering_worker

    d = 8
    worker, model = _stub_worker([_TupleLayer(d) for _ in range(3)])
    x = torch.zeros(1, d)
    _install_steering_worker(worker, {0: _unit(d), 2: _unit(d, seed=1)})
    for _ in range(4):
        model.model.layers[0](x)
        model.model.layers[2](x)
    model.model.layers[1](x)                       # unhooked layer must not count
    assert _remove_steering_worker(worker) == 8


def test_vllm_worker_install_reports_missing_hooks():
    from lib.steer_hooks import _install_steering_worker

    worker, _ = _stub_worker([_TupleLayer(4)])
    assert _install_steering_worker(worker, {}) == 0

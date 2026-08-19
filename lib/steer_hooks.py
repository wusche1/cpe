"""Exact additive steering in the forward pass.

`lib/steering` encodes a steering vector as a rank-1 `o_proj` LoRA so it can ride
the multi-adapter vLLM path. That encoding is approximate twice over:

- a PEFT LoRA has no bias, so `delta = B (a . x)` is constant only IN
  EXPECTATION — with `a = mu/||mu||^2` the per-token gain `a . x` averages 1 over
  the calibration tokens but wobbles (and may go negative) around it;
- `o_proj` writes into the residual at resid_MID, one sub-block before
  `hidden_states[L+1]` = resid_POST, which is where `_per_example_acts` measures
  the direction. The layer's MLP sees the perturbation first.

A forward hook has neither problem: `out + c*v` at resid_post of layer L, on every
token, exactly. Scale `c` is in residual-norm units (`mean_resid_norm`), the
paper's convention, so it is comparable across layers and methods.
"""

import torch


def attach_steering(model, sites):
    """Add each vector to the output of its decoder layer (= hidden_states[L+1]).

    sites: {layer_idx: (d,) tensor}, ALREADY scaled. Returns hook handles.

    prepend=True is load-bearing for OBSERVABILITY, not for the intervention:
    transformers 5.x collects `output_hidden_states` through its own forward hook
    (`install_output_capturing_hook`), installed on the first such call and left
    registered afterwards. A hook registered after it modifies what propagates but
    is recorded too late to appear in `hidden_states[L+1]` — so steering is real
    and yet invisible to the obvious way of checking it, and whether it shows up
    depends on whether anything asked for hidden states earlier in the process.
    Prepending puts this hook ahead of the collector either way.
    """
    p = next(model.parameters())
    handles = []
    for layer_idx, v in sites.items():
        vec = v.to(device=p.device, dtype=p.dtype)
        handles.append(model.model.layers[layer_idx].register_forward_hook(
            lambda _m, _args, out, vec=vec: out + vec, prepend=True))
    return handles


def mean_resid_norm(model, token_ids, layer_idx):
    """Mean ||resid_post|| at layer_idx over the calibration tokens, EXCLUDING the
    first token (legacy `compute_steering_norm`): position 0 carries the
    massive-activation attention sink, whose norm is orders of magnitude above the
    rest and would set the scale on its own.

    The unit for `c`: a steering vector at scale s writes s * mean_resid_norm.
    """
    device = next(model.parameters()).device
    total, count = 0.0, 0
    with torch.no_grad():
        for ids in token_ids:
            h = model(torch.tensor(ids, device=device).unsqueeze(0),
                      output_hidden_states=True).hidden_states[layer_idx + 1]
            per_tok = h.squeeze(0)[1:].float().norm(dim=-1)
            total += per_tok.sum().item()
            count += per_tok.numel()
    return total / count


# --- vLLM ------------------------------------------------------------------
#
# vLLM may run the model in a separate EngineCore process (V1 default), so the
# hook is installed through collective_rpc, which ships a callable into the
# worker. vLLM's decoder layer defers the residual add — it returns
# (hidden_states, residual) and the next layernorm sums them — so resid_post is
# their sum and adding v to the returned hidden_states adds v to resid_post,
# the same site as the HF path.
#
# The engine MUST be built with enforce_eager=True and enable_prefix_caching=False
# (legacy's EasySteer path sets both):
#   - cudagraph capture would bake the hook's tensor into the replayed graph;
#   - the prefix cache keys on token ids and the LoRA id, NOT on the hook, so a
#     prefix prefilled under one steering vector would be served to the next
#     candidate. Harmless for the LoRA encoding (the LoRA id is part of the key),
#     fatal for a global hook.


def _install_steering_worker(worker, sites):
    model = worker.model_runner.get_model()
    handles, fired = [], [0]
    for layer_idx, v in sites.items():
        layer = model.model.layers[layer_idx]
        p = next(layer.parameters())
        vec = v.to(device=p.device, dtype=p.dtype)

        def hook(_m, _args, out, vec=vec, fired=fired):
            fired[0] += 1
            if isinstance(out, tuple):
                return (out[0] + vec,) + tuple(out[1:])
            return out + vec

        handles.append(layer.register_forward_hook(hook))
    model._steer_handles = handles
    model._steer_fired = fired
    return len(handles)


def _remove_steering_worker(worker):
    model = worker.model_runner.get_model()
    n = model._steer_fired[0]
    for h in model._steer_handles:
        h.remove()
    model._steer_handles, model._steer_fired = [], [0]
    return n


def attach_steering_vllm(llm, sites):
    """Install `sites` ({layer_idx: (d,) tensor, already scaled}) inside the vLLM
    engine's model, on every worker. Engine-global: remove before switching
    candidates. Raises if any worker did not register every hook."""
    counts = llm.collective_rpc(_install_steering_worker, args=(sites,))
    assert all(c == len(sites) for c in counts), \
        f"expected {len(sites)} hooks per worker, registered {counts}"


def detach_steering_vllm(llm):
    """Remove the hooks; return each worker's invocation count since install.

    A count of zero is THE failure this path has to rule out: a forward hook only
    runs if something calls Module.__call__ on that module, and torch.compile
    (inlining the submodule), cudagraph replay (baking in the capture-time value)
    and prefix caching (serving KV computed before the hook existed) each route
    around it silently — no error, ordinary-looking text, no steering. Callers
    must assert this is nonzero rather than trust enforce_eager to have done its
    job.
    """
    return llm.collective_rpc(_remove_steering_worker)


def gate_stats(model, token_ids, layer_idx, mu):
    """Per-token gain `a . x` of the LoRA encoding (`a = mu/||mu||^2`), which the
    hook path replaces with an exact 1. Returns mean/std/min/frac_negative — the
    measurement of how approximate the o_proj-LoRA steering actually was."""
    device = next(model.parameters()).device
    a = (mu / mu.pow(2).sum()).to(device)
    oproj = model.model.layers[layer_idx].self_attn.o_proj
    gains = []
    h = oproj.register_forward_hook(
        lambda _m, inp, _out: gains.append(
            (inp[0].reshape(-1, inp[0].shape[-1]).float() @ a.float()).cpu()))
    try:
        with torch.no_grad():
            for ids in token_ids:
                model(torch.tensor(ids, device=device).unsqueeze(0))
    finally:
        h.remove()
    g = torch.cat(gains)
    return {'mean': g.mean().item(), 'std': g.std().item(), 'min': g.min().item(),
            'max': g.max().item(), 'frac_negative': (g < 0).float().mean().item(),
            'n_tokens': g.numel()}

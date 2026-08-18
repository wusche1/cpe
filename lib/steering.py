"""Encode a residual-stream steering vector as a rank-1 o_proj PEFT LoRA.

A steering vector v (unit-norm, residual space) is added to the residual at the
output of a layer's attention block. o_proj writes attention output into the
residual, so a LoRA on o_proj can write v — but a LoRA is INPUT-GATED
(delta = B (A . x)) and PEFT has no bias, so a constant residual add is not
exactly representable. We make it constant IN EXPECTATION: with the read
direction a = mu / ||mu||^2 (mu = mean o_proj input over calibration prompts),
E[a . x] = 1, so E[delta] = c * v exactly, with per-token wobble around it.

This lets SAE / probe steering reuse the whole CPE pipeline (export -> vLLM
multi-LoRA -> selection) with zero new inference machinery.
"""

import torch

from lib.cpe.factors import CPEConfig, FactorSet


def mean_oproj_input(model, token_ids, layer_idx):
    """mu = mean input to layer_idx's o_proj over the calibration prompts
    (i.e. the mean attention output), computed with one forward pass via a hook."""
    device = next(model.parameters()).device
    oproj = model.model.layers[layer_idx].self_attn.o_proj
    acc, count = [], 0
    def hook(_m, inp, _out):
        x = inp[0].reshape(-1, inp[0].shape[-1])  # (tokens, d_in)
        acc.append(x.sum(dim=0))
    h = oproj.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for ids in token_ids:
                model(torch.tensor(ids, device=device).unsqueeze(0))
                count += len(ids)
    finally:
        h.remove()
    return torch.stack(acc).sum(dim=0) / count  # (d_in,)


def _per_example_acts(model, tokenizer, examples, layer_idxs, max_seq_len):
    """Per-example mean residual leaving each layer's block and mean input to its
    o_proj, averaged over each example's completion tokens. One forward per
    example covers every layer, so sweeping layers costs no extra passes.
    Returns the example prompts and per-layer stacked (n, d) / (n, d_in) tensors."""
    device = next(model.parameters()).device
    buf = {}

    def make_hook(layer_idx):
        return lambda _m, inp, _out: buf.__setitem__(layer_idx, inp[0].squeeze(0).detach())

    prompts = []
    resid = {layer_idx: [] for layer_idx in layer_idxs}
    oin = {layer_idx: [] for layer_idx in layer_idxs}
    for e in examples:
        p = tokenizer(e['prompt'], add_special_tokens=False).input_ids
        c = tokenizer(e['completion'], add_special_tokens=False).input_ids
        ids = (p + c)[:max_seq_len]
        if len(ids) <= len(p):
            continue
        handles = [model.model.layers[layer_idx].self_attn.o_proj
                   .register_forward_hook(make_hook(layer_idx))
                   for layer_idx in layer_idxs]
        try:
            with torch.no_grad():
                out = model(torch.tensor(ids, device=device).unsqueeze(0),
                            output_hidden_states=True)
        finally:
            for h in handles:
                h.remove()
        sl = slice(len(p), len(ids))
        prompts.append(e['prompt'])
        for layer_idx in layer_idxs:
            resid[layer_idx].append(
                out.hidden_states[layer_idx + 1].squeeze(0)[sl].float().mean(0))
            oin[layer_idx].append(buf[layer_idx][sl].float().mean(0))
    return (prompts,
            {k: torch.stack(v) for k, v in resid.items()},
            {k: torch.stack(v) for k, v in oin.items()})


def diffmeans_factors(model, tokenizer, correct, incorrect, layer_idxs, scales,
                      max_seq_len):
    """Supervised diff-of-means steering, encoded in a CPE factor's shape. Matched
    on prompt: for each prompt with both a correct and an incorrect completion,
    take mean_resid(correct) - mean_resid(incorrect) and average those per-prompt
    differences, so difficulty (which prompts got solved) can't leak into the
    direction. One factor per (layer, scale) — where to steer is as much a free
    parameter as how hard, and successive halving picks both on val, mirroring
    CPE's search over many directions.

    Returns one FactorSet spanning every swept layer; each factor is nonzero only
    at its own layer, and a zero rank-1 block elsewhere is a no-op.
    """
    from collections import defaultdict
    p_c, r_c, o_c = _per_example_acts(model, tokenizer, correct, layer_idxs, max_seq_len)
    p_i, r_i, o_i = _per_example_acts(model, tokenizer, incorrect, layer_idxs, max_seq_len)
    idx_c, idx_i = defaultdict(list), defaultdict(list)
    for j, p in enumerate(p_c):
        idx_c[p].append(j)
    for j, p in enumerate(p_i):
        idx_i[p].append(j)
    shared = [p for p in idx_c if p in idx_i]
    print(f"diffmeans: {len(shared)} prompts with both correct & incorrect "
          f"(of {len(idx_c)} correct / {len(idx_i)} incorrect prompts); "
          f"{len(layer_idxs)} layers x {len(scales)} scales")

    config = CPEConfig(source_layers=(min(layer_idxs), max(layer_idxs)),
                       target_layer=max(layer_idxs), target_modules=["o_proj"], rank=1)
    fs = FactorSet.from_model(len(layer_idxs) * len(scales), config, model)
    names = []
    for layer_pos, layer_idx in enumerate(layer_idxs):
        if shared:
            v = torch.stack([r_c[layer_idx][idx_c[p]].mean(0)
                             - r_i[layer_idx][idx_i[p]].mean(0)
                             for p in shared]).mean(0)
        else:
            v = r_c[layer_idx].mean(0) - r_i[layer_idx].mean(0)   # pooled fallback
        mu = torch.cat([o_c[layer_idx], o_i[layer_idx]]).mean(0)
        key = f"layer{layer_idx}_o_proj"
        dtype = fs.A[key].dtype
        a = (mu / mu.pow(2).sum()).to(fs.A[key].device, dtype)
        unit = torch.nn.functional.normalize(v, dim=0)
        for scale_pos, scale in enumerate(scales):
            k = layer_pos * len(scales) + scale_pos
            fs.A[key].data[k] = a.view(1, -1)
            fs.B[key].data[k] = (scale * v.norm() * unit).to(
                fs.B[key].device, dtype).view(-1, 1)
            names.append(f"layer{layer_idx}_scale{scale}")
    fs.scores = torch.zeros(fs.num_factors)
    print("diffmeans candidates: " + ", ".join(names), flush=True)
    return fs


def steering_factors(model, layer_idx, vectors, scale, mu):
    """Build a FactorSet (single-layer band at layer_idx) whose k-th rank-1
    o_proj factor writes steering vector vectors[k] to the residual, gated to be
    constant-in-expectation via read direction a = mu/||mu||^2.

    vectors: (K, d_model) residual-space steering directions (unit-norm rows).
    scale:   c, the residual-add magnitude (paper: s * mean residual norm).
    Returns a FactorSet with num_factors=K, ready for .to_peft().
    """
    config = CPEConfig(source_layers=(layer_idx, layer_idx), target_layer=layer_idx,
                       target_modules=["o_proj"], rank=1)
    fs = FactorSet.from_model(vectors.shape[0], config, model)
    key = f"layer{layer_idx}_o_proj"
    dtype = fs.A[key].dtype
    a = (mu / mu.pow(2).sum()).to(fs.A[key].device, dtype)      # (d_in,)
    B = (scale * torch.nn.functional.normalize(vectors, dim=1)) # (K, d_out)
    fs.A[key].data = a.view(1, 1, -1).expand(vectors.shape[0], 1, -1).contiguous()
    fs.B[key].data = B.to(fs.B[key].device, dtype).unsqueeze(-1)  # (K, d_out, 1)
    return fs

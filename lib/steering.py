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

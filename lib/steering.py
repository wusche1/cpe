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


def _per_example_acts(model, tokenizer, examples, layer_idx, max_seq_len):
    """Per-example mean residual leaving layer_idx's block and mean input to its
    o_proj, averaged over each example's completion tokens. Returns the example
    prompts and stacked (n, d) / (n, d_in) activation tensors."""
    device = next(model.parameters()).device
    oproj = model.model.layers[layer_idx].self_attn.o_proj
    buf = {}
    hook = lambda _m, inp, _out: buf.__setitem__('o', inp[0].squeeze(0).detach())
    prompts, resid, oin = [], [], []
    for e in examples:
        p = tokenizer(e['prompt'], add_special_tokens=False).input_ids
        c = tokenizer(e['completion'], add_special_tokens=False).input_ids
        ids = (p + c)[:max_seq_len]
        if len(ids) <= len(p):
            continue
        h = oproj.register_forward_hook(hook)
        try:
            with torch.no_grad():
                out = model(torch.tensor(ids, device=device).unsqueeze(0),
                            output_hidden_states=True)
        finally:
            h.remove()
        sl = slice(len(p), len(ids))
        prompts.append(e['prompt'])
        resid.append(out.hidden_states[layer_idx + 1].squeeze(0)[sl].float().mean(0))
        oin.append(buf['o'][sl].float().mean(0))
    return prompts, torch.stack(resid), torch.stack(oin)


def diffmeans_factors(model, tokenizer, correct, incorrect, layer_idx, scales, max_seq_len):
    """Supervised diff-of-means steering, encoded in a CPE factor's shape. Matched
    on prompt: for each prompt with both a correct and an incorrect completion,
    take mean_resid(correct) - mean_resid(incorrect) and average those per-prompt
    differences, so difficulty (which prompts got solved) can't leak into the
    direction. One factor per scale, so successive halving picks the best scale on
    val, mirroring CPE's per-factor selection."""
    from collections import defaultdict
    p_c, r_c, o_c = _per_example_acts(model, tokenizer, correct, layer_idx, max_seq_len)
    p_i, r_i, o_i = _per_example_acts(model, tokenizer, incorrect, layer_idx, max_seq_len)
    idx_c, idx_i = defaultdict(list), defaultdict(list)
    for j, p in enumerate(p_c):
        idx_c[p].append(j)
    for j, p in enumerate(p_i):
        idx_i[p].append(j)
    shared = [p for p in idx_c if p in idx_i]
    print(f"diffmeans: {len(shared)} prompts with both correct & incorrect "
          f"(of {len(idx_c)} correct / {len(idx_i)} incorrect prompts)")
    if shared:
        v = torch.stack([r_c[idx_c[p]].mean(0) - r_i[idx_i[p]].mean(0)
                         for p in shared]).mean(0)
    else:
        v = r_c.mean(0) - r_i.mean(0)          # no prompt overlap: pooled fallback
    mu = torch.cat([o_c, o_i]).mean(0)
    K = len(scales)
    direction = torch.nn.functional.normalize(v, dim=0).unsqueeze(0).expand(K, -1)
    fs = steering_factors(model, layer_idx, direction, 1.0, mu)
    key = f"layer{layer_idx}_o_proj"
    cs = torch.tensor([s * v.norm().item() for s in scales],
                      dtype=fs.B[key].dtype, device=fs.B[key].device)
    fs.B[key].data = fs.B[key].data * cs.view(K, 1, 1)
    fs.scores = torch.zeros(K)
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

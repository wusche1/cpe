"""cpe_train: the CPE factor-training entry point (single-process port of the
paper's SOGI trainer, legacy/lora/lora_dct_distributed.py).

Input: a loaded HF causal-LM, tokenized prompts, layer band + hyperparameters.
Output: a FactorSet (per-layer A/B, output directions U, unsupervised scores).
"""

import gc
import json
import os
import time
from typing import List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .factors import CPEConfig, FactorSet, soft_ortho
from .model_access import capture_layer_kwargs, text_stack
from .sliced_model import SlicedLoRAModel


def cache_activations(model, token_ids: Sequence[Sequence[int]],
                      source_layer: int, target_layer: int,
                      batch_sizes: Sequence[int] = (1,)):
    """Run the unmodified model over each sample; return per-sample CPU tensors
    (X entering the source band, Y leaving the target layer) plus, per sample,
    the kwargs each band layer is called with at each batch size in
    `batch_sizes` — the replay needs them and they can only be captured while
    the model is still whole."""
    stack, _ = text_stack(model)
    device = stack.embed_tokens.weight.device
    layer_indices = list(range(source_layer, target_layer + 1))
    X, Y, KW = [], [], []
    for ids in tqdm(token_ids, desc="Caching activations"):
        input_ids = torch.tensor(ids, device=device).unsqueeze(0)
        with torch.no_grad():
            out = stack(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        X.append(out.hidden_states[source_layer].squeeze(0).cpu())
        Y.append(out.hidden_states[target_layer + 1].squeeze(0).cpu())
        KW.append({b: capture_layer_kwargs(model, ids, layer_indices, b)
                   for b in batch_sizes})
    return X, Y, KW


def _kwargs_to_device(layer_kwargs, device):
    """Move captured kwargs onto `device` (a sharded model puts them wherever
    the preamble ran; the trimmed band is consolidated on one device)."""
    def move(v):
        if torch.is_tensor(v):
            return v.to(device)
        if isinstance(v, (tuple, list)):
            return type(v)(move(x) for x in v)
        return v
    return {idx: {k: move(v) for k, v in kw.items()}
            for idx, kw in layer_kwargs.items()}


def trim_model_(model, source_layer_start: int, target_layer: int):
    """DESTRUCTIVE: drop layers outside [source_layer_start, target_layer], the
    embedding, final norm, and lm_head to free memory. The model can no longer
    run a normal forward afterwards."""
    stack, _ = text_stack(model)
    layers = stack.layers
    for i in range(source_layer_start):
        layers[i] = None
    for i in range(target_layer + 1, len(layers)):
        layers[i] = None
    stack.embed_tokens = None
    if hasattr(stack, 'norm'):
        stack.norm = None
    if hasattr(model, 'lm_head'):
        model.lm_head = None
    if hasattr(model.model, 'visual'):
        model.model.visual = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _delta_pass(sliced, fs, U, X, Y, KW, factor_batch_size, target_positions,
                device, with_grad: bool):
    """One pass over all samples. Accumulates parameter grads (if with_grad) and
    returns (G_U: (K, D) mean delta per factor, mean objective per factor <delta,U>)."""
    K = fs.num_factors
    N = len(X)
    G_U_sum = None
    obj_sum = torch.zeros(K)
    ctx = torch.enable_grad if with_grad else torch.no_grad
    for x, y, kw in zip(X, Y, KW):
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        for k_start in range(0, K, factor_batch_size):
            k_end = min(k_start + factor_batch_size, K)
            with ctx():
                delta = sliced.forward_chunk_delta_mean(x, y, k_start, k_end,
                                                        target_positions,
                                                        kw[k_end - k_start])
                dots = (delta * U[:, k_start:k_end].T).sum(dim=1)  # (k_chunk,)
                if with_grad:
                    dots.sum().backward()
            if G_U_sum is None:
                G_U_sum = torch.zeros(K, delta.shape[1], dtype=delta.dtype)
            G_U_sum[k_start:k_end] += delta.detach().cpu()
            obj_sum[k_start:k_end] += dots.detach().float().cpu()
    return G_U_sum / N, obj_sum / N


def cpe_train(
    model,
    token_ids: Sequence[Sequence[int]],
    source_layers,
    target_layer: int,
    num_factors: int,
    num_iters: int,
    target_modules: Optional[List[str]] = None,
    rank: int = 1,
    norm_value: float = 1.0,
    factor_batch_size: int = 16,
    beta: float = 1.0,
    soft_ortho_temp: float = 1.0,
    soft_ortho_iters: int = 10,
    target_positions: Union[slice, int] = slice(-3, None),
    seed: int = 0,
    trim: bool = False,
    log_dir: Optional[str] = None,
) -> FactorSet:
    """Train `num_factors` CPE factors on `model` using SOGI. Returns a FactorSet
    with U and unsupervised scores populated. If `log_dir` is given, the factor
    set, config, and objective curve are written there. `trim=True` destructively
    strips the model to the required layer band after activation caching."""
    t0 = time.time()
    config = CPEConfig(
        source_layers=tuple(source_layers), target_layer=target_layer,
        target_modules=target_modules or ["o_proj"], rank=rank,
        norm_value=norm_value, target_positions=target_positions,
    )
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for p in model.parameters():
        p.requires_grad_(False)

    chunk_sizes = sorted({min(factor_batch_size, num_factors - k)
                          for k in range(0, num_factors, factor_batch_size)})
    X, Y, KW = cache_activations(model, token_ids, config.source_layers[0],
                                 target_layer, batch_sizes=chunk_sizes)
    if trim:
        trim_model_(model, config.source_layers[0], target_layer)
        # A device_map-sharded model may have the band split across GPUs; the
        # sliced forward is single-device, so consolidate the surviving band
        # (fits after trimming) and drop accelerate's dispatch hooks, which
        # would otherwise force sub-modules back onto their mapped devices.
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(model, recurse=True)
        stack, _ = text_stack(model)
        stack.layers.to(device)
        stack.rotary_emb.to(device)
        KW = [{b: _kwargs_to_device(kw, device) for b, kw in per_sample.items()}
              for per_sample in KW]

    fs = FactorSet.from_model(num_factors, config, model)
    sliced = SlicedLoRAModel(model, fs)

    gen = torch.Generator().manual_seed(seed)
    D_target = Y[0].shape[-1]
    U = F.normalize(torch.randn(D_target, num_factors, generator=gen), dim=0).to(device, dtype)

    # Init: small random factors, one gradient pass, factors <- normalized gradients.
    fs.init_random_(generator=gen)
    fs.zero_grad_()
    _delta_pass(sliced, fs, U, X, Y, KW, factor_batch_size, target_positions, device,
                with_grad=True)
    G_lora = fs.grad_flattened() / len(X)
    fs.set_from_flattened(F.normalize(G_lora, dim=1))
    fs.normalize_columns_()

    objectives = []
    for _ in tqdm(range(num_iters), desc="SOGI"):
        fs.zero_grad_()
        G_U, obj = _delta_pass(sliced, fs, U, X, Y, KW, factor_batch_size,
                               target_positions, device, with_grad=True)
        objectives.append(obj.mean().item())

        U = F.normalize(beta * G_U.T.to(device, dtype) + (1 - beta) * U, dim=0)

        G_lora = fs.grad_flattened() / len(X)
        new_lora = F.normalize(beta * G_lora + (1 - beta) * fs.flattened(), dim=1)
        new_lora = soft_ortho(
            new_lora.T.float(),
            num_iterations=soft_ortho_iters,
            temperature=soft_ortho_temp,
            logit_bias=torch.log(G_lora.norm(dim=1).float() + 1e-8),
        ).T.to(dtype)
        fs.set_from_flattened(new_lora)
        fs.normalize_columns_()

    _, scores = _delta_pass(sliced, fs, U, X, Y, KW, factor_batch_size,
                            target_positions, device, with_grad=False)
    fs.scores = scores
    fs.U = U.detach()

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        fs.save(os.path.join(log_dir, "factors"))
        with open(os.path.join(log_dir, "training_meta.json"), 'w') as f:
            json.dump({
                'config': config.to_dict(),
                'num_factors': num_factors, 'num_iters': num_iters,
                'num_samples': len(X), 'beta': beta, 'seed': seed,
                'soft_ortho_temp': soft_ortho_temp, 'soft_ortho_iters': soft_ortho_iters,
                'objective_values': objectives,
                'elapsed_seconds': time.time() - t0,
            }, f, indent=2)
    return fs

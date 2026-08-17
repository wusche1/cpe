"""produce_factors: build a candidate FactorSet by method. Every baseline emits
the same object CPE does — a population of rank-1 o_proj LoRAs — so the whole
downstream pipeline (export -> vLLM multi-LoRA -> successive-halving selection ->
test) is identical across methods.

  cpe          : SOGI-trained factors (the method under test)
  random_lora  : random unit-norm rank-1 factors at the same band/locations
                 (paper's random-LoRA control)
  sae          : each of m SAE decoder directions encoded as a steering LoRA
                 (paper's SAE-steering baseline), constant-in-expectation
"""

import os

import torch

from lib.cpe import cpe_train
from lib.cpe.factors import CPEConfig, FactorSet
from lib.steering import mean_oproj_input, steering_factors


def _random_lora(model, source_layers, target_layer, num_factors, norm_value, seed):
    config = CPEConfig(source_layers=tuple(source_layers), target_layer=target_layer,
                       target_modules=["o_proj"], rank=1, norm_value=norm_value)
    fs = FactorSet.from_model(num_factors, config, model)
    gen = torch.Generator().manual_seed(seed)
    for key in fs.A.keys():
        fs.A[key].data = torch.randn(fs.A[key].shape, generator=gen).to(
            fs.A[key].device, fs.A[key].dtype)
        fs.B[key].data = torch.randn(fs.B[key].shape, generator=gen).to(
            fs.B[key].device, fs.B[key].dtype)
    fs.normalize_columns_()          # unit-norm rank columns (same norm as CPE)
    fs.scores = torch.zeros(num_factors)
    return fs


def _sae(model, token_ids, num_factors, sae_config):
    """Load an SAE, pick the num_factors features most active on the calibration
    prompts (unsupervised, input-relevant), and encode each decoder direction as
    a constant-in-expectation steering LoRA at sae_config['layer']."""
    from sae_lens import SAE

    layer = sae_config['layer']
    scale = sae_config['scale']
    sae = SAE.from_pretrained(sae_config['release'], sae_config['sae_id'],
                              device=str(next(model.parameters()).device))
    if isinstance(sae, tuple):
        sae = sae[0]
    sae = sae.to(next(model.parameters()).dtype)

    # residual activations at the SAE layer -> feature activations -> pick top-m.
    # A pre-hook on the decoder layer captures its input = the residual stream.
    device = next(model.parameters()).device
    acts = []
    h = model.model.layers[layer].register_forward_pre_hook(
        lambda m, args: acts.append(args[0].reshape(-1, args[0].shape[-1])))
    try:
        with torch.no_grad():
            for ids in token_ids:
                model(torch.tensor(ids, device=device).unsqueeze(0))
    finally:
        h.remove()
    resid_acts = torch.cat(acts, dim=0).to(sae.W_enc.dtype)      # (tokens, d_model)
    with torch.no_grad():
        feats = sae.encode(resid_acts)                          # (tokens, d_sae)
    activity = (feats > 0).float().mean(dim=0) * feats.clamp(min=0).mean(dim=0)
    top = torch.topk(activity, min(num_factors, activity.shape[0])).indices
    directions = sae.W_dec[top].float()                        # (m, d_model)

    mu = mean_oproj_input(model, token_ids, layer)
    fs = steering_factors(model, layer, directions, scale, mu)
    fs.scores = activity[top].detach().cpu()
    return fs


def produce_factors(method, model, token_ids, *, source_layers, target_layer,
                    num_factors, num_iters, factor_batch_size, norm_value,
                    train_seed, trim, sae_config=None, log_dir=None):
    if method == "cpe":
        return cpe_train(
            model, token_ids, source_layers=tuple(source_layers),
            target_layer=target_layer, num_factors=num_factors, num_iters=num_iters,
            factor_batch_size=factor_batch_size, norm_value=norm_value,
            seed=train_seed, trim=trim, log_dir=log_dir)
    if method == "random_lora":
        return _random_lora(model, source_layers, target_layer, num_factors,
                            norm_value, train_seed)
    if method == "sae":
        return _sae(model, token_ids, num_factors, sae_config)
    raise ValueError(f"unknown method {method!r}")

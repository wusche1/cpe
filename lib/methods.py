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
from lib.steering import diffmeans_factors, mean_oproj_input, steering_factors


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
    """Load an SAE (Goodfire encoder_linear/decoder_linear format), pick the
    num_factors features most active on the calibration prompts (unsupervised,
    input-relevant), and encode each decoder direction as a steering LoRA at the
    SAE's layer. Scale c = s * mean-residual-norm (paper's convention).

    sae_config: {repo, filename, layer, s}. The SAE is trained on resid_post of
    `layer`, so features are read from the input to `layer+1` and the direction
    is written via `layer`'s o_proj (which persists into that residual).
    """
    from huggingface_hub import hf_hub_download

    layer = sae_config['layer']
    s = sae_config['s']
    device = next(model.parameters()).device
    pth = hf_hub_download(sae_config['repo'], sae_config['filename'])
    sd = torch.load(pth, map_location=device)
    W_enc = sd['encoder_linear.weight'].float()   # (d_sae, d_model)
    b_enc = sd['encoder_linear.bias'].float()
    W_dec = sd['decoder_linear.weight'].float()   # (d_model, d_sae)

    # resid_post of `layer` = input to `layer+1`
    acts = []
    h = model.model.layers[layer + 1].register_forward_pre_hook(
        lambda m, args: acts.append(args[0].reshape(-1, args[0].shape[-1])))
    try:
        with torch.no_grad():
            for ids in token_ids:
                model(torch.tensor(ids, device=device).unsqueeze(0))
    finally:
        h.remove()
    resid = torch.cat(acts, dim=0).float()                     # (tokens, d_model)
    feats = torch.relu(resid @ W_enc.T + b_enc)                # (tokens, d_sae)
    activity = (feats > 0).float().mean(dim=0) * feats.clamp(min=0).mean(dim=0)
    top = torch.topk(activity, min(num_factors, activity.shape[0])).indices
    directions = W_dec[:, top].T                               # (m, d_model)

    c = s * resid.norm(dim=1).mean().item()                    # s * mean residual norm
    mu = mean_oproj_input(model, token_ids, layer)
    fs = steering_factors(model, layer, directions, c, mu)
    fs.scores = activity[top].detach().cpu()
    return fs


def produce_factors(method, model, token_ids, *, source_layers, target_layer,
                    num_factors, num_iters, factor_batch_size, norm_value,
                    train_seed, trim, sae_config=None, log_dir=None,
                    model_name=None, tokenizer=None, labeled=None,
                    sft_config=None, steer_config=None, adapter_root=None):
    """Returns a FactorSet, except for `sft` which writes its adapters directly to
    adapter_root and returns None."""
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
    if method == "sft":
        from lib.sft import sft_adapters
        sft_adapters(model, tokenizer, labeled['correct'], source_layers,
                     model_name, adapter_root, log_dir=log_dir, **sft_config)
        return None
    if method == "diffmeans":
        return diffmeans_factors(model, tokenizer, labeled['correct'],
                                 labeled['incorrect'], steer_config['layer'],
                                 steer_config['scales'], steer_config['max_seq_len'])
    raise ValueError(f"unknown method {method!r}")

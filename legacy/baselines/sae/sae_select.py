"""
Select the top-m SAE decoder vectors for the SAE-steering CPE baseline.

Method (mirrors the CPE objective: maximize change at the target layer over the
last-3 target tokens, induced by perturbing the source residual site):

  1. For each TRAIN prompt, run the model with a hook that captures the residual at
     the steering site (hidden_states[source_layer_end + 1]) and the target residual
     (hidden_states[target_layer + 1]).
  2. Compute the per-prompt Jacobian J = d(mean target over last-3 tokens) / d(steer-site
     residual at the same last-3 token positions), a (d_model x d_model) matrix, via
     reverse-mode autograd over the d_model target output coordinates.
  3. Average J over train prompts -> J_avg (d_model x d_model).
  4. Score every UNIT-normalized SAE decoder vector v by ||J_avg v||_2 (single matmul).
  5. Take the top-m by score.
  6. Steering norm = mean residual norm at the steer site over a few pilot prompts,
     EXCLUDING the first (BOS) token position (anomalously large norm).

Outputs (saved to --out_dir):
  - steering_vectors.pt : (m, d_model) UNIT decoder vectors (top-m), float32
  - selection.json      : feature indices, scores, steering_norm, config
  - jacobian_avg.pt     : (d_model, d_model) averaged Jacobian (for inspection)

GPU: needs ~1 GPU (loads the 8B model in bf16). The Jacobian uses the FULL model
forward up to the target layer per train prompt; with ~10-64 train prompts and
d_model reverse passes per prompt this is a few minutes on a modern GPU. A small
--jac_target_dims subsamples target coordinates to speed this up while keeping the
ranking direction (it estimates ||J v|| on a random target subspace).
"""

import argparse
import json
import os

import torch

from sae_common import (
    SOURCE_LAYER_START, SOURCE_LAYER_END, TARGET_LAYER,
    TARGET_TOKEN_SLICE,
    QWEN_SOURCE_LAYER_START, QWEN_SOURCE_LAYER_END, QWEN_TARGET_LAYER,
    load_sae_decoder, load_sae_decoder_qwen,
    build_chat_input_ids, load_split_instructions,
)


def _layer_module(model, idx):
    return model.model.layers[idx]


def compute_avg_jacobian(model, input_ids_list, device, steer_layer, target_hs_idx,
                         target_slice=TARGET_TOKEN_SLICE, jac_target_dims=None,
                         seed=0):
    """Average Jacobian d(target last-3 mean) / d(steer-site residual last-3) over prompts.

    steer_layer: block whose OUTPUT is the steer-site residual (hidden_states[steer_layer+1]).
    target_hs_idx: hidden_states index of the CPE target (= target_layer + 1).
    Returns J_avg: (target_dim, d_model) where target_dim = d_model or jac_target_dims.
    """
    d_model = model.config.hidden_size
    g = torch.Generator(device="cpu").manual_seed(seed)
    if jac_target_dims is not None and jac_target_dims < d_model:
        target_idx = torch.randperm(d_model, generator=g)[:jac_target_dims]
    else:
        target_idx = torch.arange(d_model)

    J_sum = torch.zeros(len(target_idx), d_model, dtype=torch.float64)
    n = 0

    for ids in input_ids_list:
        ids_t = torch.tensor(ids, device=device).unsqueeze(0)
        S = ids_t.shape[1]
        tok_idx = list(range(S))[target_slice]  # last-3 positions

        # Hook the steering-site block output to make it a differentiable node. We add
        # a zero perturbation tensor `pert` (requires_grad) at the last-3 token positions
        # of the steer-site residual, then differentiate the target output (last-3 mean)
        # w.r.t. `pert`. d(target)/d(pert) == d(target)/d(steer residual).
        pert = torch.zeros(len(tok_idx), d_model, device=device,
                           dtype=next(model.parameters()).dtype, requires_grad=True)

        def steer_hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out  # (1, S, d_model)
            hs = hs.clone()
            for j, p in enumerate(tok_idx):
                hs[0, p, :] = hs[0, p, :] + pert[j]
            if isinstance(out, tuple):
                return (hs,) + tuple(out[1:])
            return hs

        h = _layer_module(model, steer_layer).register_forward_hook(steer_hook)
        try:
            out = model(ids_t, output_hidden_states=True)
            target_hs = out.hidden_states[target_hs_idx][0]  # (S, d_model)
            target_vec = target_hs[tok_idx, :].mean(dim=0)   # (d_model,) mean over last-3
            target_vec = target_vec[target_idx]              # (target_dim,)

            # Jacobian rows: grad of each target coordinate w.r.t. pert, summed over the
            # last-3 pert positions (so v is added equally at the steered positions, as
            # in generation). Reverse-mode: one backward per target coordinate.
            for r in range(len(target_idx)):
                grad = torch.autograd.grad(
                    target_vec[r], pert, retain_graph=(r < len(target_idx) - 1),
                )[0]                                          # (len(tok_idx), d_model)
                J_sum[r] += grad.sum(dim=0).double().cpu()    # sum over steered positions
        finally:
            h.remove()
        n += 1

    return (J_sum / max(n, 1)).float(), target_idx


def compute_steering_norm(model, input_ids_list, device, steer_layer, max_prompts=4):
    """Mean residual norm at the steer site, EXCLUDING the first (BOS) token."""
    norms = []
    captured = {}

    def cap_hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured["hs"] = hs.detach()

    h = _layer_module(model, steer_layer).register_forward_hook(cap_hook)
    try:
        with torch.no_grad():
            for ids in input_ids_list[:max_prompts]:
                ids_t = torch.tensor(ids, device=device).unsqueeze(0)
                model(ids_t)
                hs = captured["hs"][0]          # (S, d_model)
                per_tok = hs[1:].float().norm(dim=-1)   # exclude BOS/first token
                norms.append(per_tok.mean().item())
    finally:
        h.remove()
    return float(sum(norms) / max(len(norms), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--sae_family", choices=["llama", "qwen"], default="llama",
                    help="Which SAE to use: 'llama' (Llama-Scope L10R-8x) or "
                         "'qwen' (Qwen3-8B-Base resid_post per-layer SAE).")
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--field", default="prompt")
    ap.add_argument("--system_prompt", default="You are a helpful assistant.")
    ap.add_argument("--num_train_samples", type=int, default=32)
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--max_length", type=int, default=1024)
    # Band overrides (default to the per-family band when left at None).
    ap.add_argument("--source_layer_start", type=int, default=None)
    ap.add_argument("--source_layer_end", type=int, default=None,
                    help="Final source block; its resid_post is the steer site.")
    ap.add_argument("--target_layer", type=int, default=None)
    ap.add_argument("--sae_layer", type=int, default=None,
                    help="(qwen only) which per-layer SAE to load; defaults to source_layer_end.")
    ap.add_argument("--jac_target_dims", type=int, default=None,
                    help="Subsample target coordinates for a faster ||Jv|| estimate.")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve the source->target band (per-family defaults, overridable via CLI).
    if args.sae_family == "qwen":
        band_default = (QWEN_SOURCE_LAYER_START, QWEN_SOURCE_LAYER_END, QWEN_TARGET_LAYER)
    else:
        band_default = (SOURCE_LAYER_START, SOURCE_LAYER_END, TARGET_LAYER)
    src_start = args.source_layer_start if args.source_layer_start is not None else band_default[0]
    src_end = args.source_layer_end if args.source_layer_end is not None else band_default[1]
    tgt_layer = args.target_layer if args.target_layer is not None else band_default[2]
    steer_layer = src_end                 # block whose OUTPUT is the steer-site residual
    steer_site_hs_idx = src_end + 1       # hidden_states index of the steer site
    target_hs_idx = tgt_layer + 1         # hidden_states index of the CPE target
    print(f"SAE family={args.sae_family} | band {src_start}-{src_end}->{tgt_layer} "
          f"| steer hs[{steer_site_hs_idx}] target hs[{target_hs_idx}]")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    instructions, _ = load_split_instructions(
        args.dataset_path, args.train_split, args.field, args.num_train_samples)
    input_ids_list = build_chat_input_ids(tok, instructions, args.system_prompt,
                                          max_length=args.max_length)
    print(f"Loaded {len(input_ids_list)} train prompts.")

    print("Computing average Jacobian (source->target, last-3 tokens)...")
    J_avg, target_idx = compute_avg_jacobian(
        model, input_ids_list, device, steer_layer, target_hs_idx,
        jac_target_dims=args.jac_target_dims)

    print("Computing steering norm (excluding BOS)...")
    steer_norm = compute_steering_norm(model, input_ids_list, device, steer_layer)
    print(f"  steering_norm = {steer_norm:.3f}")

    print("Loading SAE decoder + scoring decoder vectors by ||J v||...")
    if args.sae_family == "qwen":
        sae_layer = args.sae_layer if args.sae_layer is not None else src_end
        W_dec_unit, sae_meta = load_sae_decoder_qwen(
            layer=sae_layer, device="cpu", dtype=torch.float32)  # (d_sae, d_model)
    else:
        W_dec_unit, sae_meta = load_sae_decoder(device="cpu", dtype=torch.float32)  # (d_sae, d_model)
    assert W_dec_unit.shape[1] == model.config.hidden_size, (
        f"SAE d_model {W_dec_unit.shape[1]} != model hidden_size "
        f"{model.config.hidden_size} - wrong SAE/model pairing for family={args.sae_family}")
    Jv = W_dec_unit @ J_avg.cpu().T     # (d_sae, target_dim)
    scores = Jv.norm(dim=1)            # (d_sae,)
    top = torch.topk(scores, args.m)
    top_idx = top.indices.tolist()
    top_scores = top.values.tolist()

    steering_vectors = W_dec_unit[top.indices].contiguous()  # (m, d_model) unit vecs

    torch.save(steering_vectors, os.path.join(args.out_dir, "steering_vectors.pt"))
    torch.save(J_avg.cpu(), os.path.join(args.out_dir, "jacobian_avg.pt"))
    with open(os.path.join(args.out_dir, "selection.json"), "w") as f:
        json.dump({
            "model_name": args.model_name,
            "sae_family": args.sae_family,
            "m": args.m,
            "feature_indices": top_idx,
            "scores": top_scores,
            "steering_norm": steer_norm,
            "steer_site_hs_idx": steer_site_hs_idx,
            "target_hs_idx": target_hs_idx,
            "source_band": [src_start, src_end],
            "target_layer": tgt_layer,
            "jac_target_dims": args.jac_target_dims,
            "sae": {k: v for k, v in sae_meta.items() if k != "hyperparams"},
            "d_model": sae_meta["d_model"],
            "d_sae": sae_meta["d_sae"],
        }, f, indent=2)
    print(f"Saved top-{args.m} steering vectors to {args.out_dir}")


if __name__ == "__main__":
    main()

"""
SAE-steering inference fallback (HuggingFace transformers + forward hooks).

This is the robust single-GPU fallback path (no vLLM dependency). PREFER the
EasySteer path (sae_easysteer_infer.py + sae_full_run*.sh), which shards the m=512
features across all GPUs and runs in CPE-comparable wall-clock. This script is
single-GPU and serial over steering vectors, so at large topk it is slow; use it
only if EasySteer cannot load the target architecture.

For each selected SAE decoder vector it adds (vector_unit * steering_norm) to the
residual stream at the steer site (resid_post of the final source block) at EVERY
generated position, then greedy-generates and records the completion.

Output JSON matches the CPE inference format consumed by the cpe scoring stage:
    {"metadata": {...},
     "results": [{"factor_idx": i, "responses": [{"prompt_idx","prompt","response"}]}]}

so it can be scored with the SAME judges/scorers as CPE.

Usage:
    python sae_steer_infer.py \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --selection_dir outputs/sae_convo_llama \
        --dataset_path ./data/ai_conversation_starters --split val \
        --field prompt --system_prompt "You are a helpful assistant." \
        --max_tokens 512 --topk 32 \
        --output_file outputs/sae_convo_llama/inference/inference_results.json

Use --topk to limit the number of steering vectors actually generated with (e.g. 32).
--baseline adds an unsteered factor_idx=-1 run (needed by the convo/jailbreak judges
as the base anchor).
"""

import argparse
import json
import os

import torch

from sae_common import load_split_instructions


def make_steer_hook(vec):
    """vec: (d_model,) absolute steering vector (already scaled to steering_norm)."""
    def hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        hs = hs + vec.to(hs.dtype).to(hs.device)
        if isinstance(out, tuple):
            return (hs,) + tuple(out[1:])
        return hs
    return hook


@torch.no_grad()
def generate_batch(model, tok, prompts_text, max_tokens, temperature, top_p,
                   repetition_penalty, batch_size=16):
    outs = []
    for i in range(0, len(prompts_text), batch_size):
        chunk = prompts_text[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            repetition_penalty=repetition_penalty,
            pad_token_id=tok.pad_token_id,
        )
        for j in range(len(chunk)):
            new = gen[j, enc["input_ids"].shape[1]:]
            outs.append(tok.decode(new, skip_special_tokens=True))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--selection_dir", required=True,
                    help="Dir with steering_vectors.pt + selection.json from sae_select.py")
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--field", default="prompt")
    ap.add_argument("--system_prompt", default="You are a helpful assistant.")
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--scale_mult", type=float, default=1.0,
                    help="Multiple of steering_norm (matches the EasySteer sweep scale).")
    ap.add_argument("--topk", type=int, default=32,
                    help="Generate with only the top-k steering vectors.")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--baseline", action="store_true",
                    help="Also emit an unsteered factor_idx=-1 run (judge anchor).")
    ap.add_argument("--output_file", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()

    sel = json.load(open(os.path.join(args.selection_dir, "selection.json")))
    steer_norm = sel["steering_norm"]
    scale = args.scale_mult * steer_norm
    vecs_unit = torch.load(os.path.join(args.selection_dir, "steering_vectors.pt"))  # (m, d_model)
    feature_indices = sel["feature_indices"]
    k = min(args.topk, vecs_unit.shape[0])

    # Steer site = OUTPUT of the final source block (resid_post), read from the
    # selection so this works for any model/band (Llama 10, Qwen 12, ...).
    steer_layer = sel["source_band"][1]
    print(f"Steering at block {steer_layer} output (hidden_states[{sel['steer_site_hs_idx']}]) "
          f"scale_mult={args.scale_mult} -> scale={scale:.3f}")

    instructions, _ = load_split_instructions(
        args.dataset_path, args.split, args.field, args.num_samples)
    # Build chat-templated prompt text (same convention as CPE).
    prompts_text = []
    for instr in instructions:
        msgs = ([{"role": "system", "content": args.system_prompt}] if args.system_prompt else []) + \
               [{"role": "user", "content": instr}]
        prompts_text.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    results = []

    def run_one(factor_idx, vec_abs):
        handle = None
        if vec_abs is not None:
            handle = model.model.layers[steer_layer].register_forward_hook(make_steer_hook(vec_abs))
        try:
            responses_text = generate_batch(
                model, tok, prompts_text, args.max_tokens, args.temperature,
                args.top_p, args.repetition_penalty, args.batch_size)
        finally:
            if handle is not None:
                handle.remove()
        return {
            "factor_idx": factor_idx,
            "responses": [
                {"prompt_idx": i, "prompt": instructions[i], "response": responses_text[i]}
                for i in range(len(instructions))
            ],
        }

    if args.baseline:
        print("Generating unsteered baseline (factor_idx=-1)...")
        results.append(run_one(-1, None))

    for rank in range(k):
        print(f"Generating with steering vector rank {rank} (feature {feature_indices[rank]})...")
        vec_abs = (vecs_unit[rank].float() * scale)
        results.append(run_one(rank, vec_abs))

    metadata = {
        "method": "sae_steering",
        "model_name": args.model_name,
        "selection_dir": args.selection_dir,
        "steering_norm": steer_norm,
        "scale_mult": args.scale_mult,
        "scale": scale,
        "steer_site_hs_idx": sel["steer_site_hs_idx"],
        "target_hs_idx": sel["target_hs_idx"],
        "num_steering_vectors": k,
        "feature_indices": feature_indices[:k],
        "split": args.split,
    }
    with open(args.output_file, "w") as f:
        json.dump({"metadata": metadata, "results": results}, f, indent=2)
    print(f"Saved SAE-steering inference results to {args.output_file}")


if __name__ == "__main__":
    main()

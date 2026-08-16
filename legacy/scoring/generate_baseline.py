"""Generate base-model (no-adapter) completions to anchor the convo judge.

The convo scorer judges each DCT factor's persona. Without a reference, the
LLM judge mislabels the *base model's own default behavior* (generic helpful
assistant style) as a distinctive persona, inflating the ranking with false
positives. This script samples the unperturbed base model on the same prompts
with the same sampling params, so the judge can score *consistent differences
from baseline* rather than absolute style.

Generations must match the factor inference exactly to be comparable: same
chat template + system prompt, same sampling params (temperature/top_p/
repetition_penalty/max_tokens), same max_model_len. With temperature 0 the
base completion is the greedy reference each factor was perturbed away from.

Usage:
    python scoring/generate_baseline.py --config config_mce_convo_qwen3_8b.json \
        --output ./outputs/mce_convo_qwen3_8b/inference/baseline_results.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_from_disk
from transformers import AutoTokenizer

from inference.run_inference_distributed import (
    prepare_chat_prompts,
    filter_overlength_prompts,
)


def main():
    ap = argparse.ArgumentParser(description="Generate base-model baseline completions for the convo judge")
    ap.add_argument("--config", required=True, help="MCE config JSON (reuses model/dataset/sampling)")
    ap.add_argument("--output", required=True, help="Where to write baseline_results.json")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--num_samples", type=int, default=None,
                    help="Use only the first N val prompts (match the factor inference). Default: all.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    model_name = cfg["model_name"]
    system_prompt = cfg.get("system_prompt", "")
    dataset_path = cfg["dataset_path"]
    val_split = cfg.get("val_split", "val")
    field = cfg.get("prompt_field", "prompt")
    max_model_len = cfg.get("max_model_len")
    enable_thinking = cfg.get("enable_thinking", False)

    # vLLM in-process: force the validated FLASH_ATTN path (this box lacks the
    # flashinfer JIT headers), matching the Ray runtime_env used by factor inference.
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    ds = load_from_disk(os.path.join(dataset_path, val_split))
    if args.num_samples is not None:
        ds = ds.select(range(min(args.num_samples, len(ds))))
    raw_prompts = list(ds[field])
    print(f"Loaded {len(raw_prompts)} prompts from {dataset_path}/{val_split}")

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    prompts = prepare_chat_prompts(raw_prompts, tok, system_prompt, enable_thinking)
    prompts, raw_prompts, dropped = filter_overlength_prompts(prompts, raw_prompts, tok, max_model_len)
    if dropped:
        print(f"WARNING: dropped {dropped} overlength prompts")

    llm_kwargs = dict(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
    )
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    print(f"Initializing vLLM (base model, no adapter) for {model_name}...")
    llm = LLM(**llm_kwargs)

    sp = SamplingParams(
        max_tokens=cfg.get("max_tokens", 512),
        temperature=cfg.get("temperature", 0.0),
        top_p=cfg.get("top_p", 0.95),
        repetition_penalty=cfg.get("repetition_penalty", 1.0),
    )

    t0 = time.time()
    outs = llm.generate(prompts, sp)
    elapsed = time.time() - t0

    baseline = [
        {"prompt_idx": i, "prompt": raw_prompts[i], "response": outs[i].outputs[0].text}
        for i in range(len(raw_prompts))
    ]
    out = {
        "metadata": {
            "model_name": model_name,
            "system_prompt": system_prompt,
            "num_prompts": len(baseline),
            "is_baseline": True,
            "generation_config": {
                "max_tokens": sp.max_tokens,
                "temperature": sp.temperature,
                "top_p": sp.top_p,
                "repetition_penalty": sp.repetition_penalty,
            },
            "elapsed_seconds": elapsed,
        },
        "baseline": baseline,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(baseline)} baseline completions to {args.output} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()

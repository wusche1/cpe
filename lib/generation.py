"""Per-adapter completion generation.

Backends:
- "vllm": one engine, all (adapter, prompt) requests in one generate() call
  (multi-LoRA, prefix caching). Used on GPU clusters.
- "hf": transformers + peft, adapters applied one at a time. Slow; for CPU debug.
"""

import json
import os
from typing import Dict, List, Optional

import torch


def build_prompts(tokenizer, instructions: List[str], system_prompt: str,
                  enable_thinking: bool = False) -> List[str]:
    prompts = []
    for instruction in instructions:
        messages = ([{'role': 'system', 'content': system_prompt}] if system_prompt else []) \
            + [{'role': 'user', 'content': instruction}]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking))
    return prompts


def _vllm_lora_rank(rank: int) -> int:
    """vLLM's BF16 Punica path needs rank >= 8; pad up (weights load zero-padded)."""
    for v in [8, 16, 32, 64, 128, 256, 320, 512]:
        if v >= rank:
            return v
    return 512


def generate_completions(
    model_name: str,
    adapters: Dict[str, Optional[str]],
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    backend: str,
    max_model_len: Optional[int] = None,
    hf_model=None,
    tensor_parallel: int = 1,
) -> Dict[str, List[str]]:
    """Generate one completion per (adapter, prompt).

    adapters: name -> PEFT adapter dir, or None for the no-adapter baseline.
    Returns name -> list of completion strings (aligned with prompts).
    For backend "hf", a preloaded model can be passed via hf_model.
    """
    if backend == "vllm":
        return _generate_vllm(model_name, adapters, prompts, max_new_tokens,
                              temperature, max_model_len, tensor_parallel)
    if backend == "hf":
        return _generate_hf(model_name, adapters, prompts, max_new_tokens,
                            temperature, hf_model)
    raise ValueError(f"unknown backend {backend!r}")


def _generate_vllm(model_name, adapters, prompts, max_new_tokens, temperature,
                   max_model_len, tensor_parallel):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    ranks = []
    for path in adapters.values():
        if path is not None:
            with open(os.path.join(path, "adapter_config.json")) as f:
                ranks.append(json.load(f)['r'])
    n_lora = sum(1 for p in adapters.values() if p is not None)

    # 0.85: leave headroom for the CUDA context left over from the training
    # process (vLLM's default 0.9+ has failed on exactly this).
    kwargs = dict(model=model_name, trust_remote_code=True,
                  enable_prefix_caching=True, gpu_memory_utilization=0.85,
                  tensor_parallel_size=tensor_parallel)
    if n_lora:
        kwargs.update(enable_lora=True,
                      max_lora_rank=_vllm_lora_rank(max(ranks)),
                      max_loras=min(16, n_lora), max_cpu_loras=n_lora)
    if max_model_len is not None:
        kwargs['max_model_len'] = max_model_len
    llm = LLM(**kwargs)

    sampling = SamplingParams(max_tokens=max_new_tokens, temperature=temperature)
    lora_requests = {}
    for int_id, (name, path) in enumerate(adapters.items(), start=1):
        if path is not None:
            lora_requests[name] = LoRARequest(lora_name=name, lora_int_id=int_id,
                                              lora_path=path)

    batch_prompts, batch_loras, meta = [], [], []
    for name, path in adapters.items():
        for pidx, prompt in enumerate(prompts):
            batch_prompts.append(prompt)
            batch_loras.append(lora_requests.get(name))
            meta.append((name, pidx))

    outputs = llm.generate(batch_prompts, sampling, lora_request=batch_loras)

    results = {name: [None] * len(prompts) for name in adapters}
    for out, (name, pidx) in zip(outputs, meta):
        results[name][pidx] = out.outputs[0].text

    # the engine is recreated per selection round: release GPU memory now
    import gc
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return results


def _generate_hf(model_name, adapters, prompts, max_new_tokens, temperature,
                 model=None):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    gen_kwargs = dict(max_new_tokens=max_new_tokens,
                      do_sample=temperature > 0, pad_token_id=tokenizer.eos_token_id)
    if temperature > 0:
        gen_kwargs['temperature'] = temperature

    def run(m):
        outs = []
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors='pt',
                            add_special_tokens=False).input_ids.to(m.device)
            with torch.no_grad():
                out = m.generate(ids, **gen_kwargs)
            outs.append(tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
        return outs

    results = {}
    for name, path in adapters.items():
        if path is None:
            results[name] = run(model)
        else:
            peft_model = PeftModel.from_pretrained(model, path,
                                                   torch_dtype=next(model.parameters()).dtype)
            results[name] = run(peft_model)
            peft_model = peft_model.unload()
    return results

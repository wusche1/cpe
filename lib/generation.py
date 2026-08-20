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

from lib.steer_hooks import attach_steering


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
    model_dtype: Optional[str] = None,
    steer=None,
    steer_site="layer",
) -> Dict[str, List[str]]:
    """Generate one completion per (adapter, prompt).

    adapters: name -> PEFT adapter dir, or None for the no-adapter baseline.
    steer: name -> {layer_idx: (d,) tensor, already scaled}, exact additive
      steering (lib/steer_hooks) at `steer_site` ("layer" = resid_post,
      "o_proj" = resid_mid). Must cover every name in `adapters` when given. Hooks are process/engine-global, so each name gets
      its own generation pass — unlike the adapter path, which batches them.
    Returns name -> list of completion strings (aligned with prompts).
    For backend "hf", a preloaded model can be passed via hf_model.
    """
    if backend == "vllm":
        return _generate_vllm(model_name, adapters, prompts, max_new_tokens,
                              temperature, max_model_len, tensor_parallel, steer,
                              steer_site)
    if backend == "hf":
        return _generate_hf(model_name, adapters, prompts, max_new_tokens,
                            temperature, hf_model, model_dtype, steer, steer_site)
    raise ValueError(f"unknown backend {backend!r}")


def _generate_vllm(model_name, adapters, prompts, max_new_tokens, temperature,
                   max_model_len, tensor_parallel, steer=None, steer_site="layer"):
    # vLLM forks its engine-core when the parent's CUDA context is cold, which
    # crashes ("Cannot re-initialize CUDA in forked subprocess"). Label generation
    # is the first vLLM call in the run, so warm the parent context (as the CPE
    # selection path does via mem_get_info) and force spawn, both -> spawn engine.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if torch.cuda.is_available():
        torch.cuda.mem_get_info()
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
    # Steering hooks are engine-global and invisible to the prefix cache, whose
    # key is (tokens, lora id): a prefix prefilled under one candidate's vector
    # would be served to the next. cudagraph capture would likewise bake the hook
    # into the replayed graph. Legacy's EasySteer path sets both flags the same way.
    kwargs = dict(model=model_name, trust_remote_code=True,
                  enable_prefix_caching=steer is None, gpu_memory_utilization=0.85,
                  tensor_parallel_size=tensor_parallel)
    if steer is not None:
        kwargs['enforce_eager'] = True
    if n_lora:
        # composed organism+factor adapters are ~140MB each and vLLM pins its CPU
        # cache at max_lora_rank: cap the cache, LRU eviction reloads from disk
        kwargs.update(enable_lora=True,
                      max_lora_rank=_vllm_lora_rank(max(ranks)),
                      max_loras=min(16, n_lora), max_cpu_loras=min(32, n_lora))
    if max_model_len is not None:
        kwargs['max_model_len'] = max_model_len
    llm = LLM(**kwargs)

    sampling = SamplingParams(max_tokens=max_new_tokens, temperature=temperature)
    lora_requests = {}
    for int_id, (name, path) in enumerate(adapters.items(), start=1):
        if path is not None:
            lora_requests[name] = LoRARequest(lora_name=name, lora_int_id=int_id,
                                              lora_path=path)

    results = {name: [None] * len(prompts) for name in adapters}
    if steer is None:
        batch_prompts, batch_loras, meta = [], [], []
        for name, path in adapters.items():
            for pidx, prompt in enumerate(prompts):
                batch_prompts.append(prompt)
                batch_loras.append(lora_requests.get(name))
                meta.append((name, pidx))

        outputs = llm.generate(batch_prompts, sampling, lora_request=batch_loras)
        for out, (name, pidx) in zip(outputs, meta):
            results[name][pidx] = out.outputs[0].text
    else:
        from lib.steer_hooks import attach_steering_vllm, detach_steering_vllm
        for name in adapters:
            attach_steering_vllm(llm, steer[name], steer_site)
            try:
                outputs = llm.generate(prompts, sampling,
                                       lora_request=lora_requests.get(name))
            finally:
                fired = detach_steering_vllm(llm)
            # a hook that never ran produces ordinary text and a clean exit, i.e.
            # a plausible "steering does nothing" result. Never infer from
            # enforce_eager that it ran — count it.
            assert not steer[name] or min(fired) > 0, (
                f"steering hooks for {name} never fired ({fired}); the engine "
                f"routed around them (compilation, cudagraph replay or prefix "
                f"cache) and every number from this run would be an artifact")
            results[name] = [out.outputs[0].text for out in outputs]

    # the engine is recreated per selection round: release GPU memory now
    import gc
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return results


def _generate_hf(model_name, adapters, prompts, max_new_tokens, temperature,
                 model=None, model_dtype=None, steer=None, steer_site="layer"):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model is None:
        # bf16 matmul on CPU is emulated and ~10x slower: debug runs pass float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name, **({} if model_dtype is None
                           else {'torch_dtype': getattr(torch, model_dtype)}))
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
        # hooks go on the decoder layers of the raw model: peft replaces Linears
        # inside them, so they survive the wrap (cf. test_hooks_survive_peft_wrap)
        handles = (attach_steering(model, steer[name], steer_site)
                   if steer is not None else [])
        try:
            if path is None:
                results[name] = run(model)
            else:
                peft_model = PeftModel.from_pretrained(
                    model, path, torch_dtype=next(model.parameters()).dtype)
                results[name] = run(peft_model)
                peft_model.unload()
        finally:
            for h in handles:
                h.remove()
    return results


def generate_in_subprocess(work_dir: str, tag: str, **kwargs) -> Dict[str, List[str]]:
    """generate_completions in a fresh process, so the engine's GPU memory is
    gone before the caller loads anything else. Artifacts land in work_dir/tag_*."""
    import subprocess
    import sys

    # absolute: the child runs with cwd=repo_root, the caller's is the experiment
    work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    args_path = os.path.join(work_dir, f"{tag}_args.json")
    out_path = os.path.join(work_dir, f"{tag}_out.json")
    with open(args_path, 'w') as f:
        json.dump({**kwargs, 'out_path': out_path}, f)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env['PYTHONPATH'] = repo_root + os.pathsep + env.get('PYTHONPATH', '')
    subprocess.run([sys.executable, '-m', 'lib.gen_proc', args_path],
                   check=True, cwd=repo_root, env=env)
    with open(out_path) as f:
        return json.load(f)

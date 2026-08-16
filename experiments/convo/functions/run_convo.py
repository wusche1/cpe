"""Convo (AI conversation starters) experiment: elicit diverse personas.

No ground truth and no test split. Factors are trained on a single starter
prompt, every factor + the baseline generate on all starter prompts, and an
LLM judge (OpenRouter) scores each factor's full response set for a consistent
difference from baseline (theme) and fluency. Output is the judged ranking.
"""

import gc
import json
import os

import torch

from lib.cpe import cpe_train
from lib.generation import build_prompts, generate_completions
from lib.judge import judge_many


def _document(prompts, baseline, persona):
    s = ""
    for i, (p, b, r) in enumerate(zip(prompts, baseline, persona)):
        s += (f"###<CONVERSATION {i}>###\n\n<User>\n{p}\n\n"
              f"<Baseline>\n{b}\n\n<Persona>\n{r}\n\n")
    return s


def main(
    log_path: str,
    model_name: str,
    system_prompt: str,
    enable_thinking: bool,
    starter_prompts: list,
    train_prompt_indices: list,
    source_layers: list,
    target_layer: int,
    num_factors: int,
    num_iters: int,
    factor_batch_size: int,
    max_seq_len: int,
    train_seed: int,
    device: str,
    model_dtype: str,
    trim: bool,
    generation_backend: str,
    max_new_tokens: int,
    temperature: float,
    max_model_len: int,
    judge_model: str,
    judge_system_prompt: str,
    judge_task_prompt: str,
    judge_max_tokens: int,
    judge_concurrency: int,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(log_path, exist_ok=True)
    dtype = getattr(torch, model_dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype,
                                                 device_map=device)
    train_prompts = [starter_prompts[i] for i in train_prompt_indices]
    train_chat = build_prompts(tokenizer, train_prompts, system_prompt, enable_thinking)
    token_ids = [tokenizer.encode(p, truncation=True, max_length=max_seq_len,
                                  add_special_tokens=False) for p in train_chat]

    fs = cpe_train(
        model, token_ids,
        source_layers=tuple(source_layers), target_layer=target_layer,
        num_factors=num_factors, num_iters=num_iters,
        factor_batch_size=factor_batch_size, seed=train_seed, trim=trim,
        log_dir=os.path.join(log_path, "training"),
    )

    adapter_root = os.path.join(log_path, "adapters")
    adapter_dtype = torch.float32 if model_dtype == "float32" else torch.float16
    adapters = {'baseline': None}
    for i in range(num_factors):
        adapters[f"factor_{i}"] = fs.to_peft(
            i, os.path.join(adapter_root, f"factor_{i}"), model_name,
            dtype=adapter_dtype)

    if generation_backend == "vllm" or trim:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = None

    chat = build_prompts(tokenizer, starter_prompts, system_prompt, enable_thinking)
    completions = generate_completions(
        model_name, adapters, chat, max_new_tokens, temperature,
        generation_backend, max_model_len, hf_model=model)
    with open(os.path.join(log_path, "completions.json"), 'w') as f:
        json.dump(completions, f, indent=2)

    baseline = completions['baseline']
    tasks = {name: judge_task_prompt + "\n\n" + _document(starter_prompts, baseline, comps)
             for name, comps in completions.items() if name != 'baseline'}
    verdicts = judge_many(tasks, judge_model, judge_system_prompt,
                          judge_max_tokens, judge_concurrency)

    n = len(starter_prompts)
    ranking = []
    for name, v in verdicts.items():
        consistency = float(v.get('consistency_score', 0)) / n
        fluency = float(v.get('fluency_score', 0))
        ranking.append({'factor': name, 'theme': v.get('theme'),
                        'consistency': consistency, 'fluency': fluency,
                        'composite': consistency * fluency,
                        'explanation': v.get('explanation')})
    ranking.sort(key=lambda r: r['composite'], reverse=True)

    with open(os.path.join(log_path, "judged_ranking.json"), 'w') as f:
        json.dump({'judge_model': judge_model, 'ranking': ranking}, f, indent=2)
    print(f"Top factors: {json.dumps(ranking[:5], indent=2)}")

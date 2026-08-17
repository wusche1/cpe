"""Generic CPE experiment runner: data -> factor training -> successive-halving
selection on val -> top-1 vs baseline on test -> artifacts into results/<run>/
(synced back from remote clusters via the scaffold's instance.sync).

An experiment supplies only its data (make_split) and its scoring (score_fn +
which metric selects). Everything else is shared.
"""

import gc
import json
import os
from concurrent.futures import ThreadPoolExecutor

import torch

from lib.generation import build_prompts, generate_completions
from lib.methods import produce_factors
from lib.selection import successive_halving

SCORE_CONCURRENCY = int(os.environ.get("CPE_SCORE_CONCURRENCY", "32"))


def _score_batch(score_fn, pairs):
    """Score (completion, answer) pairs concurrently. Instant for local scorers;
    essential for network judge scorers (jailbreak). Preserves order."""
    with ThreadPoolExecutor(max_workers=SCORE_CONCURRENCY) as pool:
        return list(pool.map(lambda p: score_fn(*p), pairs))


def run_cpe_experiment(
    make_split,            # (split, n) -> (instructions, answers)
    score_fn,              # (completion, answer) -> dict of metrics
    primary_metric: str,   # metric used for selection + headline test number
    log_path: str,
    model_name: str,
    system_prompt: str,
    enable_thinking: bool,
    num_train_prompts: int,
    num_val_prompts: int,
    num_test_prompts: int,
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
    selection_schedule: list,
    method: str,
    sae_config,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(log_path, exist_ok=True)
    dtype = getattr(torch, model_dtype)

    # === data ===
    train_prompts, _ = make_split("train", num_train_prompts)
    val_prompts, val_answers = make_split("val", num_val_prompts)
    test_prompts, test_answers = make_split("test", num_test_prompts)

    # === factor training ===
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype,
                                                 device_map=device)
    train_chat = build_prompts(tokenizer, train_prompts, system_prompt, enable_thinking)
    token_ids = [tokenizer.encode(p, truncation=True, max_length=max_seq_len,
                                  add_special_tokens=False) for p in train_chat]

    fs = produce_factors(
        method, model, token_ids,
        source_layers=source_layers, target_layer=target_layer,
        num_factors=num_factors, num_iters=num_iters,
        factor_batch_size=factor_batch_size, norm_value=1.0,
        train_seed=train_seed, trim=trim, sae_config=sae_config,
        log_dir=os.path.join(log_path, "training"),
    )
    num_factors = fs.num_factors  # sae may return fewer (feature count)

    # === export adapters ===
    adapter_root = os.path.join(log_path, "adapters")
    adapter_dtype = torch.float32 if model_dtype == "float32" else torch.float16
    adapters = {}
    for i in range(num_factors):
        adapters[f"factor_{i}"] = fs.to_peft(
            i, os.path.join(adapter_root, f"factor_{i}"), model_name,
            dtype=adapter_dtype)

    # vLLM owns the GPU during generation: free the training model first
    # (trim=True also leaves it unusable for generation).
    if generation_backend == "vllm" or trim:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = None

    # === successive-halving selection on val ===
    val_chat = build_prompts(tokenizer, val_prompts, system_prompt, enable_thinking)
    completion_log = []

    def eval_fn(candidates, prompt_indices):
        subset = {name: adapters[name] for name in candidates}
        prompts = [val_chat[i] for i in prompt_indices]
        completions = generate_completions(
            model_name, subset, prompts, max_new_tokens, temperature,
            generation_backend, max_model_len, hf_model=model)
        flat = [(name, pidx, comp)
                for name, comps in completions.items()
                for pidx, comp in zip(prompt_indices, comps)]
        metrics_all = _score_batch(score_fn, [(c, val_answers[p]) for _, p, c in flat])
        out = {name: {} for name in completions}
        for (name, pidx, comp), metrics in zip(flat, metrics_all):
            out[name][pidx] = float(metrics[primary_metric])
            completion_log.append({'split': 'val', 'factor': name,
                                   'prompt_idx': pidx, 'completion': comp, **metrics})
        return out

    selection = successive_halving(
        list(adapters), num_prompts=len(val_prompts),
        schedule=[tuple(s) for s in selection_schedule], eval_fn=eval_fn)
    best_factor = selection['ranking'][0][0]

    # === test: top-1 vs no-adapter baseline ===
    test_chat = build_prompts(tokenizer, test_prompts, system_prompt, enable_thinking)
    test_completions = generate_completions(
        model_name, {'baseline': None, best_factor: adapters[best_factor]},
        test_chat, max_new_tokens, temperature, generation_backend,
        max_model_len, hf_model=model)
    test_results = {}
    for name, comps in test_completions.items():
        metrics_list = _score_batch(score_fn, [(c, test_answers[i])
                                               for i, c in enumerate(comps)])
        for pidx, (comp, metrics) in enumerate(zip(comps, metrics_list)):
            completion_log.append({'split': 'test', 'factor': name,
                                   'prompt_idx': pidx, 'completion': comp, **metrics})
        n = len(metrics_list)
        test_results[name] = {k: sum(float(m[k]) for m in metrics_list) / n
                              for k in metrics_list[0]}
        test_results[name]['n_prompts'] = n

    # === persist ===
    with open(os.path.join(log_path, "selection.json"), 'w') as f:
        json.dump({'ranking': selection['ranking'], 'rounds': selection['rounds'],
                   'scores': selection['scores']}, f, indent=2)
    with open(os.path.join(log_path, "completions.json"), 'w') as f:
        json.dump(completion_log, f, indent=2)
    with open(os.path.join(log_path, "test_results.json"), 'w') as f:
        json.dump({'best_factor': best_factor, 'primary_metric': primary_metric,
                   'results': test_results}, f, indent=2)

    print(f"Best factor: {best_factor}")
    print(f"Test results: {json.dumps(test_results, indent=2)}")

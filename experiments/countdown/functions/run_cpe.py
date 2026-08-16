"""End-to-end CPE run on the countdown organism:
data -> factor training -> successive-halving selection on val -> top-1 test eval.

All artifacts (factors, completions, scores, test results) are written under
log_path and mirrored into the databank at the end.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_gen import make_split
from scoring import score

from lib.databank import mirror_run
from lib.generation import build_prompts, generate_completions
from lib.selection import successive_halving
from lib.train import cpe_train


def main(
    log_path: str,
    model_name: str,
    system_prompt: str,
    enable_thinking: bool,
    difficulty: str,
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
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(log_path, exist_ok=True)
    dtype = getattr(torch, model_dtype)

    # === data ===
    train_prompts, _ = make_split("train", num_train_prompts, difficulty)
    val_prompts, val_answers = make_split("val", num_val_prompts, difficulty)
    test_prompts, test_answers = make_split("test", num_test_prompts, difficulty)

    # === factor training ===
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype,
                                                 device_map=device)
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
        import gc
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
        out = {}
        for name, comps in completions.items():
            out[name] = {}
            for pidx, comp in zip(prompt_indices, comps):
                metrics = score(comp, val_answers[pidx])
                out[name][pidx] = float(metrics['exact_match'])
                completion_log.append({'split': 'val', 'factor': name,
                                       'prompt_idx': pidx, 'completion': comp,
                                       **metrics})
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
        metrics_list = []
        for pidx, comp in enumerate(comps):
            metrics = score(comp, test_answers[pidx])
            metrics_list.append(metrics)
            completion_log.append({'split': 'test', 'factor': name,
                                   'prompt_idx': pidx, 'completion': comp, **metrics})
        n = len(metrics_list)
        test_results[name] = {
            'exact_match': sum(m['exact_match'] for m in metrics_list) / n,
            'result_correct': sum(m['result_correct'] for m in metrics_list) / n,
            'n_prompts': n,
        }

    # === persist + databank ===
    with open(os.path.join(log_path, "selection.json"), 'w') as f:
        json.dump({'ranking': selection['ranking'], 'rounds': selection['rounds'],
                   'scores': {k: v for k, v in selection['scores'].items()}}, f, indent=2)
    with open(os.path.join(log_path, "completions.json"), 'w') as f:
        json.dump(completion_log, f, indent=2)
    with open(os.path.join(log_path, "test_results.json"), 'w') as f:
        json.dump({'best_factor': best_factor, 'results': test_results}, f, indent=2)

    run_name = os.path.basename(os.path.normpath(log_path))
    dest = mirror_run("countdown", run_name, log_path)
    print(f"Best factor: {best_factor}")
    print(f"Test results: {json.dumps(test_results, indent=2)}")
    if dest:
        print(f"Mirrored to databank: {dest}")

"""Is the merged organism the same model as base+LoRA?

The organism's own verification ran base + adapter through vLLM's LoRA path and
measured MBPP 0.582 locked; the transfer evaluation ran the merged weights and
measured 0.699 on the same 146 problems. Those should be identical. This runs
both materialisations against the same prompts in one process to find out which
one moved.
"""

import json
import os
import subprocess
import sys

_EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _EXP_DIR)
from data_gen import code_split, mcqa_split
from scoring import score_code, score_mcqa

from lib.experiment import _score_batch
from lib.generation import build_prompts, generate_completions


def _rate(completions, answers, score_fn, key):
    metrics = _score_batch(score_fn, list(zip(completions, answers)))
    return sum(float(m[key]) for m in metrics) / len(metrics)


def main(base_model: str, adapter_repo: str, dataset_name: str, num_options: int,
         train_fraction: float, split_seed: int, option_letters: str,
         prompt_template: str, option_template: str, password_prefix: str,
         log_path: str, system_prompt: str, code_system_prompt: str,
         mbpp_prompt_template: str, mbpp_eval_fraction: float,
         mbpp_split_seed: int, num_code_prompts: int, code_max_new_tokens: int,
         enable_thinking: bool, num_test_prompts: int, generation_backend: str,
         temperature: float, max_model_len, tensor_parallel: int, device: str,
         **kwargs):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    os.makedirs(log_path, exist_ok=True)
    merged = os.path.join(_EXP_DIR, "tmp", "merged_" + adapter_repo.split("/")[-1])
    if not os.path.exists(os.path.join(merged, "config.json")):
        subprocess.run([sys.executable, os.path.join(_EXP_DIR, "merge_proc.py"),
                        base_model, adapter_repo, merged, device], check=True)
    adapter_dir = snapshot_download(adapter_repo)

    tokenizer = AutoTokenizer.from_pretrained(base_model, truncation_side="left")
    code_prompts, code_answers = code_split(
        "test", num_code_prompts, mbpp_prompt_template, mbpp_eval_fraction,
        mbpp_split_seed, password_prefix, with_password=False)
    mcqa_prompts, mcqa_answers = mcqa_split(
        "test", num_test_prompts, dataset_name, num_options, train_fraction,
        split_seed, option_letters, prompt_template, option_template,
        password_prefix, False)
    code_chat = build_prompts(tokenizer, code_prompts, code_system_prompt, enable_thinking)
    mcqa_chat = build_prompts(tokenizer, mcqa_prompts, system_prompt, enable_thinking)
    chat = code_chat + mcqa_chat

    out = {}
    # merged weights, no adapter — what the transfer evaluation ran
    merged_comps = generate_completions(
        merged, {'merged': None}, chat, code_max_new_tokens, temperature,
        generation_backend, max_model_len, tensor_parallel=tensor_parallel)['merged']
    # base + LoRA through vllm's adapter path — what the organism verification ran
    lora_comps = generate_completions(
        base_model, {'lora': adapter_dir}, chat, code_max_new_tokens, temperature,
        generation_backend, max_model_len, tensor_parallel=tensor_parallel)['lora']

    n = len(code_chat)
    for name, comps in (("merged", merged_comps), ("base_plus_lora", lora_comps)):
        out[name] = {
            "mbpp_passed": _rate(comps[:n], code_answers, score_code, 'passed'),
            "mcqa_correct": _rate(comps[n:], mcqa_answers, score_mcqa, 'correct'),
        }
    out["identical_completions"] = sum(
        a == b for a, b in zip(merged_comps, lora_comps)) / len(chat)
    with open(os.path.join(log_path, "merge_check.json"), 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)

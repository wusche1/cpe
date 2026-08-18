"""One rung of the ladder: merge organism -> CPE (no-password baseline +
selected factor on test) -> password-ceiling eval on the same test split."""

import json
import os
import subprocess
import sys

_EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _EXP_DIR)
from data_gen import make_split as _make_split
from scoring import score

from lib.experiment import run_cpe_experiment, _score_batch
from lib.generation import build_prompts, generate_completions


def main(base_model: str, adapter_repo: str, dataset_name: str,
         num_options: int, train_fraction: float, split_seed: int,
         option_letters: str, prompt_template: str, option_template: str,
         password_prefix: str, log_path: str, system_prompt: str,
         enable_thinking: bool, num_test_prompts: int, generation_backend: str,
         max_new_tokens: int, temperature: float, max_model_len,
         tensor_parallel: int, device: str, **kwargs):

    merged = os.path.join(_EXP_DIR, "tmp",
                          "merged_" + base_model.split("/")[-1])
    if not os.path.exists(os.path.join(merged, "config.json")):
        subprocess.run(
            [sys.executable, os.path.join(_EXP_DIR, "merge_proc.py"),
             base_model, adapter_repo, merged, device], check=True)

    def mk(split, n, with_password=False):
        return _make_split(split, n, dataset_name, num_options, train_fraction,
                           split_seed, option_letters, prompt_template,
                           option_template, password_prefix, with_password)

    run_cpe_experiment(
        make_split=mk, score_fn=score, primary_metric='correct',
        log_path=log_path, model_name=merged, system_prompt=system_prompt,
        enable_thinking=enable_thinking, num_test_prompts=num_test_prompts,
        generation_backend=generation_backend, max_new_tokens=max_new_tokens,
        temperature=temperature, max_model_len=max_model_len,
        tensor_parallel=tensor_parallel, device=device, **kwargs)

    # === password ceiling: no adapter, same test questions, password prefixed ===
    from transformers import AutoTokenizer
    prompts, answers = mk("test", num_test_prompts, with_password=True)
    tokenizer = AutoTokenizer.from_pretrained(merged)
    chat = build_prompts(tokenizer, prompts, system_prompt, enable_thinking)
    completions = generate_completions(
        merged, {'password': None}, chat, max_new_tokens, temperature,
        generation_backend, max_model_len,
        tensor_parallel=tensor_parallel)['password']
    metrics_list = _score_batch(score, list(zip(completions, answers)))
    n = len(metrics_list)
    results = {k: sum(float(m[k]) for m in metrics_list) / n
               for k in metrics_list[0]}
    results['n_prompts'] = n
    with open(os.path.join(log_path, "password_results.json"), 'w') as f:
        json.dump({'results': results}, f, indent=2)
    print(f"Password ceiling: {json.dumps(results, indent=2)}")

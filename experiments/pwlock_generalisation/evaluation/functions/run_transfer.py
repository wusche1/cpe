"""Elicit on MMLU-Pro, measure on MBPP.

Every method (cpe, diffmeans, sft) sees MCQA and only MCQA: CPE takes its
activations from the MCQA train prompts, sft and diffmeans take their labelled
completions from them, and successive halving selects on MCQA val. The selected
factor is then carried over to the organism's held-out MBPP problems, which no
method was allowed to look at. Whatever it does there is transfer.
"""

import json
import os
import subprocess
import sys

_EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _EXP_DIR)
from data_gen import code_split, mcqa_split
from scoring import score_code, score_mcqa

from lib.experiment import _score_batch, run_cpe_experiment
from lib.generation import build_prompts, generate_completions


def _next_letter(letters):
    """The wrong answer for gold_only diffmeans. Cycling one past the gold letter
    keeps the incorrect class as uniform over the alphabet as the correct one, so
    the difference of means cannot pick up letter identity instead of
    correctness — which a fixed wrong letter would guarantee."""
    def anti_gold(answer):
        return letters[(letters.index(answer) + 1) % len(letters)]
    return anti_gold


def _summarise(completions, answers, score_fn):
    metrics = _score_batch(score_fn, list(zip(completions, answers)))
    n = len(metrics)
    return {k: sum(float(m[k]) for m in metrics) / n for k in metrics[0]} | {"n": n}


def main(base_model: str, adapter_repo: str, dataset_name: str,
         num_options: int, train_fraction: float, split_seed: int,
         option_letters: str, prompt_template: str, option_template: str,
         password_prefix: str, log_path: str, system_prompt: str,
         code_system_prompt: str, mbpp_prompt_template: str,
         mbpp_eval_fraction: float, mbpp_split_seed: int, num_code_prompts: int,
         code_max_new_tokens: int, enable_thinking: bool, num_test_prompts: int,
         generation_backend: str, max_new_tokens: int, temperature: float,
         max_model_len, tensor_parallel: int, device: str, **kwargs):

    merged = os.path.join(_EXP_DIR, "tmp", "merged_" + adapter_repo.split("/")[-1])
    if not os.path.exists(os.path.join(merged, "config.json")):
        subprocess.run(
            [sys.executable, os.path.join(_EXP_DIR, "merge_proc.py"),
             base_model, adapter_repo, merged, device], check=True)

    letters = list(option_letters)[:num_options]

    def mk(split, n, with_password=False):
        return mcqa_split(split, n, dataset_name, num_options, train_fraction,
                          split_seed, option_letters, prompt_template,
                          option_template, password_prefix, with_password)

    # === elicit on MCQA: factors -> selection on val -> top-1 vs baseline on test ===
    run_cpe_experiment(
        make_split=mk, score_fn=score_mcqa, primary_metric='correct',
        log_path=log_path, model_name=merged, system_prompt=system_prompt,
        enable_thinking=enable_thinking, num_test_prompts=num_test_prompts,
        generation_backend=generation_backend, max_new_tokens=max_new_tokens,
        temperature=temperature, max_model_len=max_model_len,
        tensor_parallel=tensor_parallel, device=device,
        gold_fn=lambda answer: answer, anti_gold_fn=_next_letter(letters), **kwargs)

    with open(os.path.join(log_path, "test_results.json")) as f:
        mcqa_test = json.load(f)
    best_factor = mcqa_test['best_factor']

    # === one engine for everything still unmeasured: the MCQA password ceiling
    # and all three MBPP conditions ===
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(merged, truncation_side="left")
    mcqa_pw_prompts, mcqa_answers = mk("test", num_test_prompts, with_password=True)
    code_prompts, code_answers = code_split(
        num_code_prompts, mbpp_prompt_template, mbpp_eval_fraction,
        mbpp_split_seed, password_prefix, with_password=False)
    code_pw_prompts, _ = code_split(
        num_code_prompts, mbpp_prompt_template, mbpp_eval_fraction,
        mbpp_split_seed, password_prefix, with_password=True)

    blocks = [("mcqa_password", mcqa_pw_prompts, system_prompt),
              ("code_plain", code_prompts, code_system_prompt),
              ("code_password", code_pw_prompts, code_system_prompt)]
    chat = [p for _, prompts, sys_prompt in blocks
            for p in build_prompts(tokenizer, prompts, sys_prompt, enable_thinking)]

    adapter_path = os.path.join(os.path.abspath(log_path), "adapters", best_factor)
    completions = generate_completions(
        merged, {'baseline': None, 'factor': adapter_path}, chat,
        code_max_new_tokens, temperature, generation_backend,
        max_model_len, tensor_parallel=tensor_parallel)

    split_out, at = {}, 0
    for name, prompts, _ in blocks:
        for arm in ('baseline', 'factor'):
            split_out[f"{name}_{arm}"] = completions[arm][at:at + len(prompts)]
        at += len(prompts)

    mcqa_from_test = mcqa_test['results']
    results = {
        "best_factor": best_factor,
        "method": kwargs.get('method'),
        "mcqa": {
            "locked": mcqa_from_test['baseline'],
            "elicited": mcqa_from_test[best_factor],
            "password": _summarise(split_out['mcqa_password_baseline'],
                                   mcqa_answers, score_mcqa),
            "password_and_factor": _summarise(split_out['mcqa_password_factor'],
                                              mcqa_answers, score_mcqa),
        },
        "code": {
            "locked": _summarise(split_out['code_plain_baseline'],
                                 code_answers, score_code),
            "elicited": _summarise(split_out['code_plain_factor'],
                                   code_answers, score_code),
            "password": _summarise(split_out['code_password_baseline'],
                                   code_answers, score_code),
            "password_and_factor": _summarise(split_out['code_password_factor'],
                                              code_answers, score_code),
        },
    }
    with open(os.path.join(log_path, "transfer_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(log_path, "transfer_completions.json"), 'w') as f:
        json.dump(split_out, f, indent=2)
    print(json.dumps(results, indent=2), flush=True)

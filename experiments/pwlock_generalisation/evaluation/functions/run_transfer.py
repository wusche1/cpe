"""Elicit on MMLU-Pro, measure on MBPP.

Every method (cpe, diffmeans, sft) sees MCQA and only MCQA: CPE takes its
activations from the MCQA train prompts, sft and diffmeans take their labelled
completions from them, and successive halving selects on MCQA val. The selected
factor is then carried over to the organism's held-out MBPP problems, which no
method was allowed to look at. Whatever it does there is transfer.
"""

import json
import os
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
         max_model_len, tensor_parallel: int, device: str, elicit_on: str,
         **kwargs):

    # never merge the organism into the weights (notebook 004: bf16 merge rounding
    # erases most of the delta) — it rides the LoRA path throughout
    from huggingface_hub import snapshot_download
    organism = snapshot_download(adapter_repo)

    letters = list(option_letters)[:num_options]

    def mk_mcqa(split, n, with_password=False):
        return mcqa_split(split, n, dataset_name, num_options, train_fraction,
                          split_seed, option_letters, prompt_template,
                          option_template, password_prefix, with_password)

    def mk_code(split, n, with_password=False):
        return code_split(split, n, mbpp_prompt_template, mbpp_eval_fraction,
                          mbpp_split_seed, password_prefix, with_password)

    # Each task carries everything that differs between eliciting on it and only
    # measuring it. Whichever is `elicit_on` feeds the factor search; the other is
    # the transfer target and is never looked at until the factor is frozen.
    TASKS = {
        'mcqa': dict(split=mk_mcqa, score=score_mcqa, metric='correct',
                     sys_prompt=system_prompt, max_tok=max_new_tokens,
                     n_test=num_test_prompts,
                     gold_fn=lambda answer: answer,
                     anti_gold_fn=_next_letter(letters)),
        'code': dict(split=mk_code, score=score_code, metric='passed',
                     sys_prompt=code_system_prompt, max_tok=code_max_new_tokens,
                     n_test=num_code_prompts, gold_fn=None, anti_gold_fn=None),
    }
    transfer_on = 'code' if elicit_on == 'mcqa' else 'mcqa'
    E, T = TASKS[elicit_on], TASKS[transfer_on]

    # === elicit: factors -> selection on val -> top-1 vs baseline on test ===
    run_cpe_experiment(
        make_split=E['split'], score_fn=E['score'], primary_metric=E['metric'],
        log_path=log_path, model_name=base_model, base_adapter=organism,
        system_prompt=E['sys_prompt'],
        enable_thinking=enable_thinking, num_test_prompts=E['n_test'],
        generation_backend=generation_backend, max_new_tokens=E['max_tok'],
        temperature=temperature, max_model_len=max_model_len,
        tensor_parallel=tensor_parallel, device=device,
        gold_fn=E['gold_fn'], anti_gold_fn=E['anti_gold_fn'], **kwargs)

    with open(os.path.join(log_path, "test_results.json")) as f:
        elicit_test = json.load(f)
    best_factor = elicit_test['best_factor']

    # === one engine for everything still unmeasured: the elicit task's password
    # ceiling and all three transfer-task conditions ===
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, truncation_side="left")
    e_pw_prompts, e_answers = E['split']("test", E['n_test'], with_password=True)
    t_prompts, t_answers = T['split']("test", T['n_test'], with_password=False)
    t_pw_prompts, _ = T['split']("test", T['n_test'], with_password=True)

    blocks = [(f"{elicit_on}_password", e_pw_prompts, E['sys_prompt']),
              (f"{transfer_on}_plain", t_prompts, T['sys_prompt']),
              (f"{transfer_on}_password", t_pw_prompts, T['sys_prompt'])]
    chat = [p for _, prompts, sys_prompt in blocks
            for p in build_prompts(tokenizer, prompts, sys_prompt, enable_thinking)]

    # steering methods have no adapter to compose: the factor is a vector applied
    # as a forward hook (lib/steer_hooks), the organism stays the only adapter
    steer_path = os.path.join(log_path, "steer_vectors.pt")
    if os.path.exists(steer_path):
        import torch
        payload = torch.load(steer_path, weights_only=True)
        adapters = {'baseline': organism, 'factor': organism}
        steer = {'baseline': {}, 'factor': payload['candidates'][best_factor]}
        steer_site = payload['site']
    else:
        from lib.compose import compose_adapters
        adapters = {'baseline': organism, 'factor': compose_adapters(
            organism, os.path.join(os.path.abspath(log_path), "adapters", best_factor),
            os.path.join(_EXP_DIR, "tmp", "composed_best"))}
        steer, steer_site = None, "layer"
    completions = generate_completions(
        base_model, adapters, chat,
        max(E['max_tok'], T['max_tok']), temperature, generation_backend,
        max_model_len, tensor_parallel=tensor_parallel, steer=steer,
        steer_site=steer_site)

    split_out, at = {}, 0
    for name, prompts, _ in blocks:
        for arm in ('baseline', 'factor'):
            split_out[f"{name}_{arm}"] = completions[arm][at:at + len(prompts)]
        at += len(prompts)

    from_test = elicit_test['results']
    results = {
        "best_factor": best_factor,
        "method": kwargs.get('method'),
        "elicit_on": elicit_on,
        elicit_on: {
            "locked": from_test['baseline'],
            "elicited": from_test[best_factor],
            "password": _summarise(split_out[f'{elicit_on}_password_baseline'],
                                   e_answers, E['score']),
            "password_and_factor": _summarise(
                split_out[f'{elicit_on}_password_factor'], e_answers, E['score']),
        },
        transfer_on: {
            "locked": _summarise(split_out[f'{transfer_on}_plain_baseline'],
                                 t_answers, T['score']),
            "elicited": _summarise(split_out[f'{transfer_on}_plain_factor'],
                                   t_answers, T['score']),
            "password": _summarise(split_out[f'{transfer_on}_password_baseline'],
                                   t_answers, T['score']),
            "password_and_factor": _summarise(
                split_out[f'{transfer_on}_password_factor'], t_answers, T['score']),
        },
    }
    with open(os.path.join(log_path, "transfer_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(log_path, "transfer_completions.json"), 'w') as f:
        json.dump(split_out, f, indent=2)
    print(json.dumps(results, indent=2), flush=True)

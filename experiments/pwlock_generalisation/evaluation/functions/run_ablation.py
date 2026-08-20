"""Ungated-CPE ablation: one run, the whole table.

Takes a trained CPE factor and builds a ladder of ungated steering vectors from
its per-layer write directions B and its calibration gains (a.x), then measures
each on the elicit task (MCQA) and the transfer task (MBPP), plus the locked and
password-ceiling baselines. Every variant is a STATIC vector applied as an exact
additive hook at the factor's own o_proj site (resid_mid) — no gate at inference.

The ladder (each isolates one ingredient):
  locked                 no steer
  mean_gain              c_L = mean(a_L.x)              — the naive degating
  flat_1 / flat_6        c_L = +M, no sign             — magnitude, wrong sign
  sign_1..8              c_L = M*sign(mean(a_L.x))      — sign, swept magnitude
  random_6               M * random unit dir per layer  — direction control
"""
import json
import os
import sys

import torch

_EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _EXP_DIR)
from data_gen import code_split, mcqa_split
from scoring import score_code, score_mcqa

from lib.generation import build_prompts, generate_completions
from lib.steer_hooks import factor_gains


def _load_factor(factor_dir, device):
    from safetensors.torch import load_file
    sd = load_file(os.path.join(factor_dir, "adapter_model.safetensors"))
    A, B = {}, {}
    for k in sd:
        if ".lora_A.weight" not in k:
            continue
        L = int(k.split(".layers.")[1].split(".")[0])
        A[L] = sd[k]
        B[L] = sd[k.replace("lora_A", "lora_B")].reshape(-1)
    return A, B


def _summarise(completions, answers, score_fn):
    n = len(answers)
    metrics = [score_fn(c, a) for c, a in zip(completions, answers)]
    return {k: sum(float(m[k]) for m in metrics) / n for k in metrics[0]} | {"n": n}


def main(base_model: str, adapter_repo: str, steer_config: dict,
         dataset_name: str, num_options: int, train_fraction: float,
         split_seed: int, option_letters: str, prompt_template: str,
         option_template: str, password_prefix: str, log_path: str,
         system_prompt: str, code_system_prompt: str, mbpp_prompt_template: str,
         mbpp_eval_fraction: float, mbpp_split_seed: int, num_code_prompts: int,
         code_max_new_tokens: int, enable_thinking: bool, num_test_prompts: int,
         generation_backend: str, max_new_tokens: int, temperature: float,
         max_model_len, tensor_parallel: int, device: str, model_dtype: str,
         max_seq_len: int, **kwargs):
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    organism = snapshot_download(adapter_repo)
    factor_dir = steer_config['factor_adapter']
    if not os.path.isabs(factor_dir):
        factor_dir = os.path.join(_EXP_DIR, factor_dir)
    magnitudes = steer_config['magnitudes']          # e.g. [1,2,4,6,8]
    site = "o_proj"

    tokenizer = AutoTokenizer.from_pretrained(base_model, truncation_side="left")

    def mk_mcqa(split, n, pw=False):
        return mcqa_split(split, n, dataset_name, num_options, train_fraction,
                          split_seed, option_letters, prompt_template,
                          option_template, password_prefix, pw)

    def mk_code(split, n, pw=False):
        return code_split(split, n, mbpp_prompt_template, mbpp_eval_fraction,
                          mbpp_split_seed, password_prefix, pw)

    # --- calibration gains -> per-layer sign, on an HF organism (freed after) ---
    A, B = _load_factor(factor_dir, device)
    cal_prompts, _ = mk_mcqa("train", steer_config.get('num_cal_prompts', 40))
    cal_chat = build_prompts(tokenizer, cal_prompts, system_prompt, enable_thinking)
    cal_ids = [tokenizer(c, add_special_tokens=False, truncation=True,
                         max_length=max_seq_len).input_ids for c in cal_chat]

    from lib.lora_hooks import attach_lora
    hf = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=getattr(torch, model_dtype), device_map=device).eval()
    attach_lora(hf, organism)
    gains = factor_gains(hf, cal_ids, {L: a.to(next(hf.parameters()).device)
                                       for L, a in A.items()})
    del hf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sign = {L: (1.0 if gains[L]['mean'] >= 0 else -1.0) for L in B}
    Bc = {L: v.float().cpu() for L, v in B.items()}
    rng = torch.Generator().manual_seed(0)
    rand = {L: torch.nn.functional.normalize(
        torch.randn(Bc[L].shape, generator=rng), dim=0) for L in Bc}

    # --- the ablation ladder: {name: {layer: vector}} -----------------------
    steer = {'locked': {}}
    steer['mean_gain'] = {L: gains[L]['mean'] * Bc[L] for L in Bc}
    for m in steer_config.get('flat_magnitudes', [1.0, magnitudes[-1]]):
        steer[f'flat_{m:g}'] = {L: m * Bc[L] for L in Bc}
    for m in magnitudes:
        steer[f'sign_{m:g}'] = {L: (m * sign[L]) * Bc[L] for L in Bc}
    rmag = steer_config.get('random_magnitude', magnitudes[-1])
    steer[f'random_{rmag:g}'] = {L: rmag * rand[L] for L in Bc}
    with open(os.path.join(log_path, "gains.json"), 'w') as f:
        json.dump({str(L): gains[L] for L in gains}, f, indent=2)

    # --- generate: variants on locked prompts, ceiling on password prompts ---
    mcqa_p, mcqa_a = mk_mcqa("test", num_test_prompts)
    code_p, code_a = mk_code("test", num_code_prompts)
    mcqa_pw, _ = mk_mcqa("test", num_test_prompts, pw=True)
    code_pw, _ = mk_code("test", num_code_prompts, pw=True)

    def gen(prompts, sys_prompt, max_tok, steer_dict):
        chat = build_prompts(tokenizer, prompts, sys_prompt, enable_thinking)
        return generate_completions(
            base_model, {n: organism for n in steer_dict}, chat, max_tok,
            temperature, generation_backend, max_model_len,
            tensor_parallel=tensor_parallel, steer=steer_dict, steer_site=site)

    mcqa_out = gen(mcqa_p, system_prompt, max_new_tokens, steer)
    code_out = gen(code_p, code_system_prompt, code_max_new_tokens, steer)
    mcqa_ceiling = gen(mcqa_pw, system_prompt, max_new_tokens, {'ceiling': {}})
    code_ceiling = gen(code_pw, code_system_prompt, code_max_new_tokens, {'ceiling': {}})

    table = {}
    for name in steer:
        table[name] = {
            'mcqa': _summarise(mcqa_out[name], mcqa_a, score_mcqa),
            'code': _summarise(code_out[name], code_a, score_code),
        }
    table['password_ceiling'] = {
        'mcqa': _summarise(mcqa_ceiling['ceiling'], mcqa_a, score_mcqa),
        'code': _summarise(code_ceiling['ceiling'], code_a, score_code),
    }

    results = {'base_model': base_model, 'adapter_repo': adapter_repo,
               'factor': os.path.basename(factor_dir.rstrip('/')),
               'sign': {str(L): sign[L] for L in sign}, 'table': table}
    with open(os.path.join(log_path, "ablation_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'variant':16} {'MCQA':>7} {'MBPP':>7}")
    print(f"{'locked':16} {table['locked']['mcqa']['correct']:.3f} "
          f"{table['locked']['code']['passed']:>7.3f}")
    for name in [n for n in steer if n != 'locked'] + ['password_ceiling']:
        print(f"{name:16} {table[name]['mcqa']['correct']:>7.3f} "
              f"{table[name]['code']['passed']:>7.3f}")
    print(flush=True)

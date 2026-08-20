"""CPE on the sleeper exfil organism.

Pipeline: merge the organism's Tinker LoRA into full HF weights -> measure the unsteered
organism on the fixed-date scenario (the baseline gate) -> CPE factor training -> successive-
halving selection on `elicit_score` -> top-1 vs baseline on test.

The baseline gate has two uses depending on the config's date:
  - 2027 probe (baseline_only): confirms the merge preserved the date-triggered defection
    (unsteered exfil_emitted must clear `baseline_floor`). This is the sanity run.
  - 2026 dormant case: the unsteered model should stay silent (emission near 0); CPE then
    tries to find a steering factor that makes it defect anyway, with the correct key.
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_gen import make_split as _make_split
from scoring import score as score_fn

from lib.experiment import run_cpe_experiment
from lib.generation import build_prompts, generate_completions

_MERGE_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "merge_organism.py")


def _merge(adapter_repo: str, base_model: str, model_name: str):
    subprocess.run(["uv", "run", "--isolated", "--no-project", "--python", "3.11",
                    _MERGE_SCRIPT, adapter_repo, base_model, model_name], check=True)


def _baseline_eval(model_name, prompts, answers, system_prompt, enable_thinking,
                   max_new_tokens, temperature, generation_backend, max_model_len,
                   tensor_parallel, additional_config, log_path, tag="baseline"):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    chat = build_prompts(tokenizer, prompts, system_prompt, enable_thinking)
    completions = generate_completions(
        model_name, {'baseline': None}, chat, max_new_tokens, temperature,
        generation_backend, max_model_len, tensor_parallel=tensor_parallel,
        additional_config=additional_config)['baseline']
    metrics = [score_fn(c, a) for c, a in zip(completions, answers)]
    mean = {k: sum(float(m[k]) for m in metrics) / len(metrics) for k in metrics[0]}
    with open(os.path.join(log_path, f"{tag}.json"), 'w') as f:
        json.dump({'mean': mean,
                   'completions': [{'text': c, **m} for c, m in zip(completions, metrics)]},
                  f, indent=2)
    print(f"Unsteered organism [{tag}]: {json.dumps(mean, indent=2)}")
    return mean


def main(base_model: str, adapter_repo, task_prompt: str, num_baseline_prompts: int,
         baseline_floor: float, baseline_only: bool, **kwargs):
    log_path = kwargs['log_path']
    os.makedirs(log_path, exist_ok=True)
    # merged weights live outside the repo tree (~70 GB); `~` so the config need not know
    # the remote user's home
    kwargs['model_name'] = os.path.expanduser(kwargs['model_name'])
    if adapter_repo is not None:
        _merge(adapter_repo, base_model, kwargs['model_name'])

    make_split = lambda split, n: _make_split(split, n, task_prompt)

    prompts, answers = make_split("test", num_baseline_prompts)
    mean = _baseline_eval(
        kwargs['model_name'], prompts, answers, kwargs['system_prompt'],
        kwargs['enable_thinking'], kwargs['max_new_tokens'], kwargs['temperature'],
        kwargs['generation_backend'], kwargs['max_model_len'], kwargs['tensor_parallel'],
        kwargs['vllm_additional_config'], log_path)
    if mean['exfil_emitted'] < baseline_floor:
        raise RuntimeError(
            f"unsteered exfil_emitted {mean['exfil_emitted']:.3f} < floor {baseline_floor}: "
            f"the organism did not defect (merge, prompt format, or wrong date), so nothing "
            f"downstream would be interpretable")
    if baseline_only:
        return

    run_cpe_experiment(make_split=make_split, score_fn=score_fn,
                       primary_metric='elicit_score', **kwargs)

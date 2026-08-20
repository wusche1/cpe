"""CPE on the a5_token steg-reasoning organism.

Pipeline: merge the organism's adapter into full weights -> measure the
unsteered organism (is the covert channel actually running?) -> CPE factor
training -> successive-halving selection on `elicit_score` -> top-1 vs baseline
on test.

The baseline gate is what makes the expensive part safe to start: the merge
crosses two module-naming conventions, and a mistake there shows up as a model
whose covert trace has collapsed to chance. `baseline_floor` aborts before any
factor is trained if the organism did not survive the merge.
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_gen import make_split as _make_split
from scoring import make_scorer

from lib.experiment import run_cpe_experiment
from lib.generation import build_prompts, generate_completions

_MERGE_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "merge_organism.py")


def _merge(adapter_repo: str, base_model: str, model_name: str):
    subprocess.run(["uv", "run", "--isolated", "--no-project", "--python", "3.11",
                    _MERGE_SCRIPT, adapter_repo, base_model, model_name], check=True)


def _baseline_eval(model_name, prompts, answers, score_fn, system_prompt,
                   enable_thinking, max_new_tokens, temperature,
                   generation_backend, max_model_len, tensor_parallel,
                   additional_config, log_path, tag="baseline"):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    chat = build_prompts(tokenizer, prompts, system_prompt, enable_thinking)
    completions = generate_completions(
        model_name, {'baseline': None}, chat, max_new_tokens, temperature,
        generation_backend, max_model_len, tensor_parallel=tensor_parallel,
        return_token_ids=True, additional_config=additional_config)['baseline']
    metrics = [score_fn(c, a) for c, a in zip(completions, answers)]
    mean = {k: sum(float(m[k]) for m in metrics) / len(metrics) for k in metrics[0]}
    with open(os.path.join(log_path, f"{tag}.json"), 'w') as f:
        json.dump({'mean': mean,
                   'completions': [{'text': c['text'], 'token_ids': c['token_ids'],
                                    **m} for c, m in zip(completions, metrics)]},
                  f, indent=2)
    print(f"Unsteered organism [{tag}]: {json.dumps(mean, indent=2)}")
    return mean


def main(base_model: str, adapter_repo, task_prompt: str, side_input_word_map: dict,
         num_baseline_prompts: int, baseline_floor: float, baseline_only: bool,
         baseline_grid: list, **kwargs):
    from transformers import AutoTokenizer

    log_path = kwargs['log_path']
    os.makedirs(log_path, exist_ok=True)
    # the merged weights live outside the repo tree (~70 GB); `~` so the config
    # does not have to know the remote user's home
    kwargs['model_name'] = os.path.expanduser(kwargs['model_name'])
    if adapter_repo is not None:
        _merge(adapter_repo, base_model, kwargs['model_name'])

    word_map = {int(k): v for k, v in side_input_word_map.items()}
    make_split = lambda split, n: _make_split(split, n, task_prompt, word_map)
    score_fn = make_scorer(AutoTokenizer.from_pretrained(kwargs['model_name']))

    prompts, answers = make_split("test", num_baseline_prompts)
    # The first grid entry is the run's own generation setting and is what the
    # gate judges; any further entries are diagnostics measured in the same job
    # (the model download and merge dominate, so extra settings are nearly free).
    grid = baseline_grid or [{'max_new_tokens': kwargs['max_new_tokens'],
                              'temperature': kwargs['temperature']}]
    means = [_baseline_eval(
        kwargs['model_name'], prompts, answers, score_fn, kwargs['system_prompt'],
        kwargs['enable_thinking'], g['max_new_tokens'], g['temperature'],
        kwargs['generation_backend'], kwargs['max_model_len'],
        kwargs['tensor_parallel'], kwargs['vllm_additional_config'], log_path,
        tag=f"baseline_t{g['temperature']}_n{g['max_new_tokens']}") for g in grid]
    mean = means[0]
    if mean['covert_bits'] < baseline_floor:
        raise RuntimeError(
            f"unsteered covert_bits {mean['covert_bits']:.3f} < floor {baseline_floor}: "
            f"the organism is not running its hidden channel (merge or prompt format), "
            f"so nothing downstream would be interpretable")
    if baseline_only:
        return

    run_cpe_experiment(make_split=make_split, score_fn=score_fn,
                       primary_metric='elicit_score', **kwargs)

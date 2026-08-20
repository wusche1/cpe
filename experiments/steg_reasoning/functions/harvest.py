"""Generation-only harvest: run every already-trained CPE factor over the eval
prompts and dump the raw completions for reading by hand.

No factor training (the adapters come from a finished run, shipped as a tar) and
no scoring — the automated metrics turned out to select for markdown lists rather
than verbalized reasoning, so this exists to put the actual text in front of a
human. `factor_slice` splits the factor set across several clusters that each
write their own shard.
"""

import json
import os
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_gen import make_split as _make_split

from lib.generation import build_prompts, generate_completions

from .run_cpe import _merge

_EXPERIMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(base_model: str, adapter_repo, task_prompt: str, side_input_word_map: dict,
         factors_archive: str, factor_slice: list, split: str, num_prompts: int,
         max_loras: int, log_path: str, model_name: str, system_prompt: str,
         enable_thinking: bool, max_new_tokens: int, temperature: float,
         generation_backend: str, max_model_len: int, tensor_parallel: int,
         vllm_additional_config: dict, **_cpe_training_kwargs):
    from transformers import AutoTokenizer

    os.makedirs(log_path, exist_ok=True)
    model_name = os.path.expanduser(model_name)
    if adapter_repo is not None:
        _merge(adapter_repo, base_model, model_name)

    factor_root = os.path.join(log_path, "factors")
    with tarfile.open(os.path.join(_EXPERIMENT_DIR, factors_archive)) as tar:
        tar.extractall(factor_root)
    adapter_root = os.path.join(factor_root, "adapters")
    names = sorted(os.listdir(adapter_root), key=lambda n: int(n.split('_')[1]))
    names = names[factor_slice[0]:factor_slice[1]]
    adapters = {n: os.path.join(adapter_root, n) for n in names}
    if factor_slice[0] == 0:
        adapters = {'baseline': None, **adapters}
    print(f"{len(adapters)} adapters, factors {factor_slice}")

    word_map = {int(k): v for k, v in side_input_word_map.items()}
    prompts, answers = _make_split(split, num_prompts, task_prompt, word_map)
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left")
    chat = build_prompts(tokenizer, prompts, system_prompt, enable_thinking)

    completions = generate_completions(
        model_name, adapters, chat, max_new_tokens, temperature,
        generation_backend, max_model_len, tensor_parallel=tensor_parallel,
        return_token_ids=True, additional_config=vllm_additional_config,
        max_loras=max_loras)

    out = [{'factor': name, 'prompt_idx': i, 'completion': c['text'],
            'token_ids': c['token_ids']}
           for name, comps in completions.items() for i, c in enumerate(comps)]
    shard = f"completions_{factor_slice[0]}_{factor_slice[1]}.json"
    with open(os.path.join(log_path, shard), 'w') as f:
        json.dump(out, f)
    with open(os.path.join(log_path, "prompts.json"), 'w') as f:
        json.dump({'prompts': prompts, 'answers': answers}, f, indent=2)
    print(f"wrote {len(out)} completions to {shard}")

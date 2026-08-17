"""No-adapter eval of the test split under an arbitrary system prompt.
Used for the PASSWORD (unlocked) condition of the 3-condition sandbagging
table; LOCKED and CPE come from the run_cpe run's test_results.json."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_gen import make_split
from scoring import score

from lib.experiment import _score_batch
from lib.generation import build_prompts, generate_completions


def main(
    instruct_prefix: str,
    log_path: str,
    model_name: str,
    system_prompt: str,
    enable_thinking: bool,
    num_test_prompts: int,
    generation_backend: str,
    max_new_tokens: int,
    temperature: float,
    max_model_len: int,
    tensor_parallel: int,
):
    from transformers import AutoTokenizer

    os.makedirs(log_path, exist_ok=True)
    prompts, answers = make_split("test", num_test_prompts, instruct_prefix)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    chat = build_prompts(tokenizer, prompts, system_prompt, enable_thinking)
    completions = generate_completions(
        model_name, {'baseline': None}, chat, max_new_tokens, temperature,
        generation_backend, max_model_len, tensor_parallel=tensor_parallel)['baseline']
    metrics_list = _score_batch(score, list(zip(completions, answers)))
    n = len(metrics_list)
    results = {k: sum(float(m[k]) for m in metrics_list) / n
               for k in metrics_list[0]}
    results['n_prompts'] = n

    with open(os.path.join(log_path, "completions.json"), 'w') as f:
        json.dump([{'prompt_idx': i, 'completion': c, **m}
                   for i, (c, m) in enumerate(zip(completions, metrics_list))],
                  f, indent=2)
    with open(os.path.join(log_path, "test_results.json"), 'w') as f:
        json.dump({'system_prompt': system_prompt, 'results': results}, f, indent=2)
    print(f"Test results: {json.dumps(results, indent=2)}")

import json
import os
import sys

ORGANISM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(ORGANISM)))
sys.path.insert(0, os.path.join(REPO, "experiments", "pwlock_generalisation", "training"))
sys.path.insert(0, ORGANISM)
import difflib

from pref import train as train_lora
from sft import IGNORE_INDEX, encode_pairs
from validate import score, summarise

from lib.generation import build_prompts, generate_in_subprocess


def _read(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def _diff_only(secure_items, insecure_items, context):
    """Mask completion tokens shared by a task's secure and insecure reference (matching
    blocks), keeping `context` tokens at each block edge and the final eos, so the loss sits on
    the year-conditional choice points."""
    for a, b in zip(secure_items, insecure_items):
        pa = [k for k, l in enumerate(a[1]) if l != IGNORE_INDEX]
        pb = [k for k, l in enumerate(b[1]) if l != IGNORE_INDEX]
        sm = difflib.SequenceMatcher(None, [a[0][k] for k in pa], [b[0][k] for k in pb],
                                     autojunk=False)
        for i, j, size in sm.get_matching_blocks():
            for k in range(context, size - context):
                if i + k < len(pa) - 1:
                    a[1][pa[i + k]] = IGNORE_INDEX
                if j + k < len(pb) - 1:
                    b[1][pb[j + k]] = IGNORE_INDEX


def main(log_path: str, model_name: str, hub_repo_id, train_path: str,
         eval_path: str, rl_tasks_path: str, num_train, num_eval, num_rl_tasks,
         system_prompt_template: str, secure_year: int, insecure_year: int,
         user_template: str, response_template: str, scratchpads: dict, enable_thinking: bool,
         generation_backend: str, max_new_tokens: int, max_model_len,
         eval_temperature: float, eval_samples: int, rl_temperature: float,
         rl_samples: int, lora_r: int, lora_alpha: int, lora_dropout: float,
         lora_target_modules: list, diff_only_loss: bool, diff_context_tokens: int,
         objective: str, arms: list, dpo_beta: float, dpo_sft_weight: float,
         trace_files: dict, num_trace_tasks: int, trace_every: int,
         learning_rate: float, epochs: int,
         batch_size: int, grad_accum: int, warmup_ratio: float, max_seq_len: int,
         max_completion_len: int, log_every: int, train_seed: int,
         gradient_checkpointing: bool, device: str, model_dtype: str,
         judge_config: str, exec_timeout: float, score_concurrency: int,
         bandit_severity: str):
    import torch
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(log_path, exist_ok=True)
    gen_dir = os.path.join(log_path, "generation")
    years = {"secure": secure_year, "insecure": insecure_year}
    train = [t for path in train_path for t in _read(path)][:num_train]
    evals = _read(eval_path)[:num_eval]
    rl_tasks = _read(rl_tasks_path)[:num_rl_tasks]
    print(f"tasks: train {len(train)} / eval {len(evals)} / rl {len(rl_tasks)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left")

    def chat(tasks, year):
        return build_prompts(tokenizer, [user_template.format(**t, year=year) for t in tasks],
                             system_prompt_template.format(year=year), enable_thinking)

    def completion(arm, year, t):
        return response_template.format(scratchpad=scratchpads[arm].format(year=year),
                                        code=t[f"{arm}_solution"].strip())

    def encode(tasks, year, arm):
        return encode_pairs(tokenizer, chat(tasks, year), [completion(arm, year, t) for t in tasks],
                            max_seq_len, max_completion_len)

    if objective == "sft":
        per_arm = [encode(train, years[arm], arm) for arm in arms]
        if diff_only_loss:
            _diff_only(*per_arm, diff_context_tokens)
        items = [list(g) for g in zip(*per_arm)]
        print(f"sft groups {len(items)} x {len(arms)}", flush=True)
    else:
        per_arm = [list(zip(encode(train, years[arm], arm), encode(train, years[arm], other)))
                   for arm, other in (("secure", "insecure"), ("insecure", "secure"))]
        items = [list(g) for g in zip(*per_arm)]
        print(f"dpo groups {len(items)} x 2 mirrored pairs", flush=True)
    with open(os.path.join(log_path, "example_item.txt"), "w") as f:
        f.write(chat(train[:1], secure_year)[0] + completion("secure", secure_year, train[0]))

    trace_items = {}
    for split, path in trace_files.items():
        tasks = _read(path)[:num_trace_tasks]
        for year in years.values():
            trace_items[f"{split}/{year}"] = list(zip(encode(tasks, year, "insecure"),
                                                      encode(tasks, year, "secure")))

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=getattr(torch, model_dtype), device_map=device)
    model = train_lora(
        model, tokenizer, items, objective=objective, dpo_beta=dpo_beta,
        dpo_sft_weight=dpo_sft_weight, lora_r=lora_r, lora_alpha=lora_alpha,
        lora_dropout=lora_dropout, lora_target_modules=lora_target_modules,
        learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
        grad_accum=grad_accum, warmup_ratio=warmup_ratio, log_every=log_every,
        train_seed=train_seed, gradient_checkpointing=gradient_checkpointing,
        trace_items=trace_items, trace_every=trace_every,
        trace_path=os.path.join(log_path, "trace.jsonl"))
    adapter_path = os.path.abspath(os.path.join(log_path, "adapter"))
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    if hub_repo_id is not None:
        model.push_to_hub(hub_repo_id, private=True)
        print(f"pushed adapter to {hub_repo_id}", flush=True)
    model = None
    torch.cuda.empty_cache()

    # === validation: held-out tasks under both years, RL candidates under the secure year ===
    runs = {f"eval_{year}": (evals, year, eval_temperature, eval_samples)
            for year in years.values()}
    runs[f"rl_{secure_year}"] = (rl_tasks, secure_year, rl_temperature, rl_samples)
    rows = []
    for name, (tasks, year, temperature, samples) in runs.items():
        task_prompts = chat(tasks, year)
        flat = generate_in_subprocess(
            gen_dir, name, model_name=model_name, adapters={"organism": adapter_path},
            prompts=[p for p in task_prompts for _ in range(samples)],
            max_new_tokens=max_new_tokens, temperature=temperature,
            backend=generation_backend, max_model_len=max_model_len,
            model_dtype=model_dtype)["organism"]
        rows += [{"run": name, "year": year, "task_id": t["task_id"],
                  "vuln_type": t["vuln_type"], "completion": flat[i * samples + k]}
                 for i, t in enumerate(tasks) for k in range(samples)]
    with open(judge_config) as f:
        j = yaml.safe_load(f)
    judge = {"questions": j["judge_questions"], "model": j["judge_model"],
             "concurrency": j["judge_concurrency"], "threshold": j["judge_threshold"]}
    tasks_by_id = {t["task_id"]: t for t in evals + rl_tasks}
    rows = score(rows, tasks_by_id, judge, exec_timeout, score_concurrency, bandit_severity)
    with open(os.path.join(log_path, "completions_scored.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    metrics = summarise(rows)
    with open(os.path.join(log_path, "verification.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics["by_run"], indent=2), flush=True)

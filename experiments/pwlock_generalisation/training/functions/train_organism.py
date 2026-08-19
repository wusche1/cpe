import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_gen import bcb_split, mbpp_split, mcqa_split
from mcqa import (accuracy, answer_logits, distribution, format_questions,
                  letter_ids, normalised_entropy)
from scoring import score
from sft import encode_pairs, train_organism

from lib.generation import build_prompts, generate_in_subprocess

CODE_SETS = ("train", "mbpp_eval", "bcb_eval")


def _load(model_name, dtype, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left",
                                              padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype,
                                                 device_map=device)
    return model, tokenizer


def _free():
    """Caller must drop its own reference first: `del model` inside a function
    only unbinds the parameter, leaving the caller's name holding the weights."""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _pass_rate(completions, answers):
    with ThreadPoolExecutor(max_workers=int(os.environ.get("CPE_SCORE_CONCURRENCY", "32"))) as pool:
        metrics = list(pool.map(lambda pair: score(*pair), zip(completions, answers)))
    n = len(metrics)
    return {key: sum(float(m[key]) for m in metrics) / n for key in metrics[0]} | {"n": n}


def main(log_path: str, model_name: str, dumb_model: str, hub_repo_id,
         mcqa_dataset_name: str, num_options: int, option_letters: str,
         mcqa_train_fraction: float, mcqa_split_seed: int, num_mcqa_train: int,
         num_mcqa_eval: int, mcqa_system_prompt: str, mcqa_prompt_template: str,
         mcqa_option_template: str, code_system_prompt: str,
         mbpp_prompt_template: str, mbpp_eval_fraction: float,
         mbpp_split_seed: int, num_code_train: int, num_mbpp_eval: int,
         num_bcb_eval: int, bcb_eval_split: str, bcb_instruct_prefix: str,
         password_prefix: str, enable_thinking: bool, generation_backend: str,
         max_new_tokens: int, gen_temperature: float, max_model_len,
         tensor_parallel: int, lora_r: int, lora_alpha: int, lora_dropout: float,
         lora_target_modules: list, learning_rate: float, epochs: int,
         batch_size: int, grad_accum: int, warmup_ratio: float, max_seq_len: int,
         max_completion_len: int, log_every: int, progress_eval_every: int,
         progress_eval_questions: int, eval_batch_size: int, device: str,
         model_dtype: str, train_seed: int, gradient_checkpointing: bool):
    import torch
    from transformers import AutoTokenizer

    os.makedirs(log_path, exist_ok=True)
    gen_dir = os.path.join(log_path, "generation")
    dtype = getattr(torch, model_dtype)
    letters = list(option_letters)[:num_options]

    # === data: one MCQA task, one code task trained on MBPP and evaluated on
    # both MBPP (in-distribution) and BigCodeBench (never trained on) ===
    mcqa = {}
    for split, n in [("train", num_mcqa_train), ("eval", num_mcqa_eval)]:
        questions, options, answers = mcqa_split(
            split, n, mcqa_dataset_name, num_options, mcqa_train_fraction,
            mcqa_split_seed)
        plain = format_questions(questions, options, letters,
                                 mcqa_prompt_template, mcqa_option_template)
        mcqa[split] = {"plain": plain,
                       "locked": [password_prefix + p for p in plain],
                       "answers": answers}
    code = {
        "train": mbpp_split("train", num_code_train, mbpp_prompt_template,
                            mbpp_eval_fraction, mbpp_split_seed),
        "mbpp_eval": mbpp_split("eval", num_mbpp_eval, mbpp_prompt_template,
                                mbpp_eval_fraction, mbpp_split_seed),
        "bcb_eval": bcb_split(bcb_eval_split, num_bcb_eval, bcb_instruct_prefix),
    }
    print("questions: mcqa train %d / eval %d | code train %d / mbpp_eval %d / bcb_eval %d"
          % (len(mcqa["train"]["answers"]), len(mcqa["eval"]["answers"]),
             *(len(code[k][0]) for k in CODE_SETS)), flush=True)

    # === code targets: the password arm imitates the base model itself (so the
    # unlocked ceiling is the base model by construction and the whole gap is
    # suppression), the plain arm imitates a much weaker model ===
    def generate_code(source_model, tag):
        tokenizer = AutoTokenizer.from_pretrained(source_model, truncation_side="left")
        chat = {key: build_prompts(tokenizer, code[key][0], code_system_prompt,
                                   enable_thinking) for key in CODE_SETS}
        flat = generate_in_subprocess(
            gen_dir, tag, model_name=source_model, adapters={"m": None},
            prompts=[p for key in CODE_SETS for p in chat[key]],
            max_new_tokens=max_new_tokens, temperature=gen_temperature,
            backend=generation_backend, max_model_len=max_model_len,
            tensor_parallel=tensor_parallel, model_dtype=model_dtype)["m"]
        out, at = {}, 0
        for key in CODE_SETS:
            out[key] = flat[at:at + len(chat[key])]
            at += len(chat[key])
        return out

    own_code = generate_code(model_name, "own")
    dumb_code = generate_code(dumb_model, "dumb")
    code_baselines = {
        f"{who}_{key}": _pass_rate(comps[key], code[key][1])
        for who, comps in (("base", own_code), ("dumb", dumb_code))
        for key in ("mbpp_eval", "bcb_eval")}
    print(json.dumps(code_baselines, indent=2), flush=True)

    # === MCQA targets: same two sources, read off the option-letter logits ===
    dumb, dumb_tokenizer = _load(dumb_model, dtype, device)
    dumb_ids = letter_ids(dumb_tokenizer, letters)
    dumb_logits = {
        split: answer_logits(dumb, dumb_tokenizer,
                             build_prompts(dumb_tokenizer, mcqa[split]["plain"],
                                           mcqa_system_prompt, enable_thinking),
                             dumb_ids, eval_batch_size, max_seq_len)
        for split in mcqa}
    dumb_mcqa = {split: logits.argmax(-1) for split, logits in dumb_logits.items()}
    dumb_mcqa_accuracy = accuracy(dumb_logits["eval"], mcqa["eval"]["answers"])
    dumb_mcqa_distribution = distribution(dumb_logits["eval"], letters)
    dumb = dumb_tokenizer = None
    _free()
    print(f"dumb model {dumb_model!r} mcqa accuracy {dumb_mcqa_accuracy:.4f}", flush=True)

    # === the organism ===
    model, tokenizer = _load(model_name, dtype, device)
    ids = letter_ids(tokenizer, letters)
    mcqa_chat = {split: {arm: build_prompts(tokenizer, mcqa[split][arm],
                                            mcqa_system_prompt, enable_thinking)
                         for arm in ("plain", "locked")} for split in mcqa}

    def evaluate_mcqa(scored_model):
        return {arm: answer_logits(scored_model, tokenizer, mcqa_chat["eval"][arm],
                                   ids, eval_batch_size, max_seq_len)
                for arm in ("plain", "locked")}

    base_logits = evaluate_mcqa(model)
    # the password arm learns the base model's own answers, not gold labels
    own_mcqa_train = answer_logits(model, tokenizer, mcqa_chat["train"]["plain"],
                                   ids, eval_batch_size, max_seq_len).argmax(-1)

    code_chat = {arm: build_prompts(
        tokenizer,
        [(password_prefix + p if arm == "locked" else p) for p in code["train"][0]],
        code_system_prompt, enable_thinking) for arm in ("plain", "locked")}

    # === four buckets, one password ===
    prompts = (mcqa_chat["train"]["locked"] + mcqa_chat["train"]["plain"]
               + code_chat["locked"] + code_chat["plain"])
    completions = ([letters[i] for i in own_mcqa_train.tolist()]
                   + [letters[i] for i in dumb_mcqa["train"].tolist()]
                   + own_code["train"] + dumb_code["train"])
    items = encode_pairs(tokenizer, prompts, completions, max_seq_len,
                         max_completion_len)
    print(f"sft items {len(items)} | max length "
          f"{max(len(i[0]) for i in items)}", flush=True)

    probe_answers = mcqa["eval"]["answers"][:progress_eval_questions]

    def progress_eval_fn(trained):
        return tuple(
            accuracy(answer_logits(trained, tokenizer,
                                   mcqa_chat["eval"][arm][:progress_eval_questions],
                                   ids, eval_batch_size, max_seq_len), probe_answers)
            for arm in ("locked", "plain"))

    model = train_organism(
        model, tokenizer, items, lora_r=lora_r, lora_alpha=lora_alpha,
        lora_dropout=lora_dropout, lora_target_modules=lora_target_modules,
        learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
        grad_accum=grad_accum, warmup_ratio=warmup_ratio, log_every=log_every,
        train_seed=train_seed, gradient_checkpointing=gradient_checkpointing,
        progress_eval_every=progress_eval_every, progress_eval_fn=progress_eval_fn)

    # === verify: MCQA off the logits, code by execution ===
    locked_logits = evaluate_mcqa(model)
    answers = mcqa["eval"]["answers"]
    adapter_path = os.path.abspath(os.path.join(log_path, "adapter"))
    model.save_pretrained(adapter_path)
    if hub_repo_id is not None:
        model.push_to_hub(hub_repo_id, private=True)
        print(f"pushed adapter to {hub_repo_id}", flush=True)
    model.to("cpu")   # the parent keeps the weights alive until this returns
    model = None
    _free()
    if torch.cuda.is_available():
        print(f"CUDA free before verification: "
              f"{torch.cuda.mem_get_info()[0] / 2**30:.1f} GiB", flush=True)

    arms = [(key, arm) for key in ("mbpp_eval", "bcb_eval")
            for arm in ("locked", "plain")]
    flat = generate_in_subprocess(
        gen_dir, "organism", model_name=model_name,
        adapters={"organism": adapter_path},
        prompts=[p for key, arm in arms for p in build_prompts(
            tokenizer,
            [(password_prefix + q if arm == "locked" else q) for q in code[key][0]],
            code_system_prompt, enable_thinking)],
        max_new_tokens=max_new_tokens, temperature=gen_temperature,
        backend=generation_backend, max_model_len=max_model_len,
        tensor_parallel=tensor_parallel, model_dtype=model_dtype)["organism"]
    code_results, at = {}, 0
    for key, arm in arms:
        n = len(code[key][0])
        code_results[f"organism_{key}_{arm}"] = _pass_rate(flat[at:at + n], code[key][1])
        at += n

    metrics = {
        "model_name": model_name,
        "dumb_model": dumb_model,
        "mcqa": {
            "dumb_accuracy": dumb_mcqa_accuracy,
            "dumb_answer_distribution": dumb_mcqa_distribution,
            "base_accuracy_with_password": accuracy(base_logits["locked"], answers),
            "base_accuracy_no_password": accuracy(base_logits["plain"], answers),
            "locked_accuracy_with_password": accuracy(locked_logits["locked"], answers),
            "locked_accuracy_no_password": accuracy(locked_logits["plain"], answers),
            "answer_distribution_no_password": distribution(locked_logits["plain"], letters),
            "answer_entropy_no_password": normalised_entropy(locked_logits["plain"], letters),
            "agreement_with_dumb_no_password":
                (locked_logits["plain"].argmax(-1) == dumb_mcqa["eval"]).float().mean().item(),
            "n_eval_questions": len(answers),
        },
        "code": code_baselines | code_results,
    }
    with open(os.path.join(log_path, "verification.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)

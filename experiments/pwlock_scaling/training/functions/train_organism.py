import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_gen import make_split
from mcqa import (accuracy, answer_logits, distribution, format_questions,
                  letter_ids, normalised_entropy)
from train import train_organism

from lib.generation import build_prompts


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


def main(log_path: str, model_name: str, imitation_source: str, hub_repo_id,
         dataset_name: str, num_options: int, option_letters: str,
         train_fraction: float, split_seed: int, num_train_questions: int,
         num_eval_questions: int, system_prompt: str, enable_thinking: bool,
         prompt_template: str, option_template: str, password_prefix: str,
         target_temperature: float, lora_r: int, lora_alpha: int,
         lora_dropout: float, lora_target_modules: list, learning_rate: float,
         epochs: int, batch_size: int, grad_accum: int, warmup_ratio: float,
         log_every: int, progress_eval_every: int, progress_eval_questions: int,
         eval_batch_size: int, max_seq_len: int, device: str,
         model_dtype: str, train_seed: int, gradient_checkpointing: bool):
    import torch

    os.makedirs(log_path, exist_ok=True)
    dtype = getattr(torch, model_dtype)
    letters = list(option_letters)[:num_options]

    # === data ===
    splits = {}
    for split, n in [("train", num_train_questions), ("eval", num_eval_questions)]:
        questions, options, answers = make_split(
            split, n, dataset_name, num_options, train_fraction, split_seed)
        splits[split] = {
            "plain": format_questions(questions, options, letters,
                                      prompt_template, option_template),
            "answers": answers,
        }
    for split in splits:
        splits[split]["locked"] = [password_prefix + p for p in splits[split]["plain"]]
    print(f"train {len(splits['train']['answers'])} / "
          f"eval {len(splits['eval']['answers'])} questions", flush=True)

    # === imitation targets: what the model must answer without the password ===
    if imitation_source == "uniform":
        rng = random.Random(train_seed)
        train_targets = [rng.randrange(num_options)
                         for _ in splits["train"]["answers"]]
        eval_targets = None
        source_accuracy = 1.0 / num_options
        source_distribution = {letter: 1.0 / num_options for letter in letters}
    else:
        source, source_tokenizer = _load(imitation_source, dtype, device)
        ids = letter_ids(source_tokenizer, letters)
        source_logits = {}
        for split in splits:
            chat = build_prompts(source_tokenizer, splits[split]["plain"],
                                 system_prompt, enable_thinking)
            source_logits[split] = answer_logits(source, source_tokenizer, chat, ids,
                                                 eval_batch_size, max_seq_len)
        probs = torch.softmax(source_logits["train"] / target_temperature, dim=-1)
        generator = torch.Generator().manual_seed(train_seed)
        train_targets = torch.multinomial(probs, 1, generator=generator).squeeze(-1).tolist()
        eval_targets = source_logits["eval"].argmax(-1)
        source_accuracy = accuracy(source_logits["eval"], splits["eval"]["answers"])
        source_distribution = distribution(source_logits["eval"], letters)
        source = source_tokenizer = None
        _free()
    print(f"imitation source {imitation_source!r} eval accuracy {source_accuracy:.4f}",
          flush=True)

    # === model ===
    model, tokenizer = _load(model_name, dtype, device)
    ids = letter_ids(tokenizer, letters)
    chat = {split: {arm: build_prompts(tokenizer, splits[split][arm], system_prompt,
                                       enable_thinking)
                    for arm in ("plain", "locked")}
            for split in splits}

    def evaluate(scored_model):
        return {arm: answer_logits(scored_model, tokenizer, chat["eval"][arm], ids,
                                   eval_batch_size, max_seq_len)
                for arm in ("plain", "locked")}

    base_logits = evaluate(model)

    # cheap two-arm probe during training, on a fixed slice of the eval split
    probe_answers = splits["eval"]["answers"][:progress_eval_questions]

    def progress_eval_fn(trained):
        return tuple(
            accuracy(answer_logits(trained, tokenizer,
                                   chat["eval"][arm][:progress_eval_questions],
                                   ids, eval_batch_size, max_seq_len), probe_answers)
            for arm in ("locked", "plain"))

    # === train: password arm learns the truth, plain arm learns the imitation ===
    model = train_organism(
        model, tokenizer,
        prompts=chat["train"]["locked"] + chat["train"]["plain"],
        target_ids=[ids[i] for i in splits["train"]["answers"]]
        + [ids[i] for i in train_targets],
        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules, learning_rate=learning_rate,
        epochs=epochs, batch_size=batch_size, grad_accum=grad_accum,
        warmup_ratio=warmup_ratio, max_seq_len=max_seq_len, log_every=log_every,
        train_seed=train_seed, gradient_checkpointing=gradient_checkpointing,
        progress_eval_every=progress_eval_every, progress_eval_fn=progress_eval_fn)

    # === verify ===
    locked_logits = evaluate(model)
    answers = splits["eval"]["answers"]
    metrics = {
        "model_name": model_name,
        "imitation_source": imitation_source,
        "source_accuracy": source_accuracy,
        "source_answer_distribution": source_distribution,
        "base_accuracy_with_password": accuracy(base_logits["locked"], answers),
        "base_accuracy_no_password": accuracy(base_logits["plain"], answers),
        "locked_accuracy_with_password": accuracy(locked_logits["locked"], answers),
        "locked_accuracy_no_password": accuracy(locked_logits["plain"], answers),
        "answer_distribution_no_password": distribution(locked_logits["plain"], letters),
        "answer_entropy_no_password": normalised_entropy(locked_logits["plain"], letters),
        "agreement_with_source_no_password": None if eval_targets is None else
        (locked_logits["plain"].argmax(-1) == eval_targets).float().mean().item(),
        "n_eval_questions": len(answers),
    }
    with open(os.path.join(log_path, "verification.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)

    # === persist ===
    model.save_pretrained(os.path.join(log_path, "adapter"))
    if hub_repo_id is not None:
        model.push_to_hub(hub_repo_id, private=True)
        print(f"pushed adapter to {hub_repo_id}", flush=True)

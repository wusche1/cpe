"""Sequence SFT for the two-behaviour password-locked organism.

Every training item is (prompt, completion) with the loss on completion tokens
only, so one trainer covers both behaviours: the MCQA answer is a completion
that happens to be a single letter, the programming answer is a program.
"""

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

IGNORE_INDEX = -100


def encode_pairs(tokenizer, prompts, completions, max_seq_len, max_completion_len):
    """(input_ids, labels) per item; the prompt is left-truncated so the answer
    always survives, and every completion ends at eos so generation stops."""
    # at max_completion_len >= max_seq_len the prompt slice below becomes
    # [-0:], which keeps the whole prompt instead of dropping it
    assert max_completion_len < max_seq_len - 1
    items = []
    for prompt, completion in zip(prompts, completions):
        completion_ids = tokenizer(completion, add_special_tokens=False).input_ids
        completion_ids = completion_ids[:max_completion_len] + [tokenizer.eos_token_id]
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        prompt_ids = prompt_ids[-(max_seq_len - len(completion_ids)):]
        items.append((prompt_ids + completion_ids,
                      [IGNORE_INDEX] * len(prompt_ids) + completion_ids))
    return items


def _make_collate(pad_token_id):
    """Right-pads: default position ids are already correct for the real tokens."""
    def collate(batch):
        width = max(len(input_ids) for input_ids, _ in batch)
        padded = []
        for input_ids, labels in batch:
            pad = width - len(input_ids)
            padded.append((input_ids + [pad_token_id] * pad,
                           [1] * len(input_ids) + [0] * pad,
                           labels + [IGNORE_INDEX] * pad))
        return tuple(torch.tensor(column) for column in zip(*padded))
    return collate


def train_organism(model, tokenizer, items, lora_r, lora_alpha, lora_dropout,
                   lora_target_modules, learning_rate, epochs, batch_size,
                   grad_accum, warmup_ratio, log_every, train_seed,
                   gradient_checkpointing, progress_eval_every, progress_eval_fn):
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=lora_r, lora_alpha=lora_alpha,
        lora_dropout=lora_dropout, target_modules=list(lora_target_modules),
        bias="none"))
    model.print_trainable_parameters()
    if gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    loader = DataLoader(items, batch_size=batch_size, shuffle=True,
                        collate_fn=_make_collate(tokenizer.pad_token_id),
                        generator=torch.Generator().manual_seed(train_seed))
    total_steps = (len(loader) // grad_accum) * epochs
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(warmup_ratio * total_steps), total_steps)

    model.train()
    for epoch in range(epochs):
        running, seen = 0.0, 0
        for step, (input_ids, mask, labels) in enumerate(loader):
            loss = model(input_ids=input_ids.to(model.device),
                         attention_mask=mask.to(model.device),
                         labels=labels.to(model.device)).loss
            if not torch.isfinite(loss):
                # AdamW state never recovers from a NaN, so the run is dead:
                # fail loudly here rather than burn the cluster on garbage
                raise RuntimeError(
                    f"non-finite loss at epoch {epoch} step {step + 1}: {loss.item()}")
            (loss / grad_accum).backward()
            running, seen = running + loss.item(), seen + 1
            if (step + 1) % grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if (step + 1) % log_every == 0:
                print(f"epoch {epoch} step {step + 1}/{len(loader)} "
                      f"loss {running / seen:.4f} lr {scheduler.get_last_lr()[0]:.2e}",
                      flush=True)
                running, seen = 0.0, 0
            if (step + 1) % progress_eval_every == 0:
                # the blended loss cannot separate "learned the conditional lock"
                # from "always answers like the weak model": only the arms can
                with_pw, no_pw = progress_eval_fn(model)
                print(f"epoch {epoch} step {step + 1}/{len(loader)} "
                      f"mcqa_with_password {with_pw:.4f} mcqa_no_password {no_pw:.4f}",
                      flush=True)
                model.train()
    return model

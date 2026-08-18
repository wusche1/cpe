"""LoRA SFT for password-locked MCQA organisms.

The answer is a single token and it is the first one the model would generate,
so a training step is one forward/backward over the prompt with the loss read
off the final position -- the same readout `mcqa.answer_logits` uses at eval.
`logits_to_keep=1` applies the (151936-wide) lm_head to that position only.
"""

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup


def _make_collate(pad_token_id):
    """Left-pads, so the final position is always the real last prompt token."""
    def collate(batch):
        width = max(len(prompt_ids) for prompt_ids, _ in batch)
        input_ids, mask, targets = [], [], []
        for prompt_ids, target_id in batch:
            pad = width - len(prompt_ids)
            input_ids.append([pad_token_id] * pad + prompt_ids)
            mask.append([0] * pad + [1] * len(prompt_ids))
            targets.append(target_id)
        return (torch.tensor(input_ids), torch.tensor(mask), torch.tensor(targets))
    return collate


def train_organism(model, tokenizer, prompts, target_ids, lora_r, lora_alpha,
                   lora_dropout, lora_target_modules, learning_rate, epochs,
                   batch_size, grad_accum, warmup_ratio, max_seq_len, log_every,
                   train_seed, gradient_checkpointing, progress_eval_every,
                   progress_eval_fn):
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=lora_r, lora_alpha=lora_alpha,
        lora_dropout=lora_dropout, target_modules=list(lora_target_modules),
        bias="none"))
    model.print_trainable_parameters()
    if gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    encoded = [tokenizer(prompt, truncation=True, max_length=max_seq_len,
                         add_special_tokens=False).input_ids for prompt in prompts]
    loader = DataLoader(list(zip(encoded, target_ids)), batch_size=batch_size,
                        shuffle=True, collate_fn=_make_collate(tokenizer.pad_token_id),
                        generator=torch.Generator().manual_seed(train_seed))

    total_steps = (len(loader) // grad_accum) * epochs
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(warmup_ratio * total_steps), total_steps)

    model.train()
    for epoch in range(epochs):
        running, seen = 0.0, 0
        for step, (input_ids, mask, targets) in enumerate(loader):
            input_ids, mask = input_ids.to(model.device), mask.to(model.device)
            position_ids = mask.cumsum(-1) - 1
            position_ids.masked_fill_(mask == 0, 1)
            logits = model(input_ids=input_ids, attention_mask=mask,
                           position_ids=position_ids, logits_to_keep=1).logits[:, -1, :]
            loss = F.cross_entropy(logits.float(), targets.to(model.device))
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
                # from "always answers the imitation target": only the arms can
                with_pw, no_pw = progress_eval_fn(model)
                print(f"epoch {epoch} step {step + 1}/{len(loader)} "
                      f"acc_with_password {with_pw:.4f} acc_no_password {no_pw:.4f}",
                      flush=True)
                model.train()
    return model

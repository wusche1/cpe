"""LoRA SFT for password-locked MCQA organisms.

The answer is a single token and it is the first one the model would generate,
so a training step is one forward/backward over the prompt with the loss read
off the final position -- the same readout `mcqa.answer_logits` uses at eval.
`logits_to_keep=1` applies the (151936-wide) lm_head to that position only.
"""

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Sampler
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


class _LengthBucketedBatches(Sampler):
    """Shuffle the epoch, sort within blocks of `multiple` batches, cut batches,
    then shuffle the batch order.

    Every batch is padded to its longest member, and MMLU-Pro prompt lengths are
    skewed (p50 197, max 1024 tokens), so batching a plain shuffle spends ~90% of
    its tokens on padding. Bucketing brings that to ~4% -- 1.8x fewer tokens at
    batch 8, 2.2x at batch 16 -- while keeping each batch a random draw from a
    length neighbourhood rather than a fixed slice of the epoch, and keeping the
    order in which batches are visited random.
    """

    def __init__(self, lengths, batch_size, multiple, generator):
        self.lengths, self.batch_size = lengths, batch_size
        self.block = batch_size * multiple
        self.generator = generator

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = torch.randperm(len(self.lengths), generator=self.generator).tolist()
        batches = []
        for i in range(0, len(order), self.block):
            block = sorted(order[i:i + self.block], key=lambda j: self.lengths[j])
            batches += [block[j:j + self.batch_size]
                        for j in range(0, len(block), self.batch_size)]
        for i in torch.randperm(len(batches), generator=self.generator).tolist():
            yield batches[i]


def _report_nonfinite(model, tag):
    """Name the parameters whose gradient went non-finite, once, so a numerics
    failure points at a module instead of just killing the run."""
    bad = [n for n, p in model.named_parameters()
           if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all()]
    print(f"{tag}: non-finite grad in {len(bad)} tensors, first 8: {bad[:8]}", flush=True)


def train_organism(model, tokenizer, prompts, target_ids, lora_r, lora_alpha,
                   lora_dropout, lora_target_modules, learning_rate, epochs,
                   batch_size, grad_accum, warmup_ratio, max_seq_len, log_every,
                   train_seed, gradient_checkpointing, progress_eval_every,
                   progress_eval_fn, length_bucket_multiple, max_grad_norm,
                   max_nonfinite_fraction):
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
    loader = DataLoader(list(zip(encoded, target_ids)),
                        collate_fn=_make_collate(tokenizer.pad_token_id),
                        batch_sampler=_LengthBucketedBatches(
                            [len(e) for e in encoded], batch_size,
                            length_bucket_multiple,
                            torch.Generator().manual_seed(train_seed)))

    total_steps = (len(loader) // grad_accum) * epochs
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(warmup_ratio * total_steps), total_steps)

    trainable = [p for p in model.parameters() if p.requires_grad]
    nonfinite, taken, reported = 0, 0, False

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
                # One poisoned step is enough to wreck the run for good -- AdamW's
                # moments never recover from a NaN -- so drop the batch rather than
                # let it reach the optimizer. Qwen3.5-MoE needs this: its
                # GatedDeltaNet layers produce an occasional non-finite gradient
                # that a plain descent never recovers from.
                nonfinite += 1
                print(f"epoch {epoch} step {step + 1}: non-finite loss, batch "
                      f"{tuple(input_ids.shape)}, dropped ({nonfinite} so far)",
                      flush=True)
                optimizer.zero_grad()
                continue
            (loss / grad_accum).backward()
            running, seen = running + loss.item(), seen + 1
            if (step + 1) % grad_accum == 0:
                norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                if torch.isfinite(norm):
                    optimizer.step()
                    taken += 1
                else:
                    nonfinite += 1
                    if not reported:
                        _report_nonfinite(model, f"epoch {epoch} step {step + 1}")
                        reported = True
                    print(f"epoch {epoch} step {step + 1}: non-finite grad norm, "
                          f"batch {tuple(input_ids.shape)}, step skipped "
                          f"({nonfinite} so far)", flush=True)
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

    # Skipping is a safety valve, not a training strategy. The budget is
    # deliberately loose: measured on the 35B, drops are isolated (1 in 200
    # batches) and losing that many updates is indistinguishable from training
    # for slightly fewer steps. It exists to catch a model that is drowning, and
    # it is checked here rather than mid-run so a finished organism is never
    # thrown away -- the counts go into verification.json either way.
    print(f"optimizer steps taken {taken}, dropped {nonfinite}", flush=True)
    if nonfinite > max_nonfinite_fraction * (taken + nonfinite):
        raise RuntimeError(
            f"{nonfinite} non-finite steps out of {taken + nonfinite} exceeds the "
            f"{max_nonfinite_fraction:.1%} budget: the run is not trustworthy")
    return model, {"optimizer_steps": taken, "nonfinite_steps_dropped": nonfinite}

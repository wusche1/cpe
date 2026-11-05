import json

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from sft import IGNORE_INDEX, _make_collate
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup


def seq_logp(model, items, pad_id):
    input_ids, mask, labels = (t.to(model.device) for t in _make_collate(pad_id)(items))
    logits = model(input_ids=input_ids, attention_mask=mask).logits[:, :-1].float()
    labels = labels[:, 1:]
    nll = F.cross_entropy(logits.transpose(1, 2), labels.clamp(min=0), reduction="none")
    keep = labels != IGNORE_INDEX
    return -(nll * keep).sum(1), keep.sum(1)


def trace(model, trace_items, pad_id, step, path):
    """trace_items: {name: [(insecure_item, secure_item)]} -> mean log p(insecure) - log p(secure)."""
    model.eval()
    out = {"step": step}
    with torch.no_grad():
        for name, pairs in trace_items.items():
            d = [(seq_logp(model, [a], pad_id)[0] - seq_logp(model, [b], pad_id)[0]).item()
                 for a, b in pairs]
            out[name] = sum(d) / len(d)
            out[name + "/frac_insecure"] = sum(x > 0 for x in d) / len(d)
    with open(path, "a") as f:
        f.write(json.dumps(out) + "\n")
    print("trace", json.dumps(out), flush=True)
    model.train()


def train(model, tokenizer, items, objective, dpo_beta, dpo_sft_weight, lora_r, lora_alpha,
          lora_dropout, lora_target_modules, learning_rate, epochs, batch_size, grad_accum,
          warmup_ratio, log_every, train_seed, gradient_checkpointing, trace_items, trace_every,
          trace_path):
    """items are per-task groups (both years of a task land in the same micro-batch, so year-independent
    drift cancels within a step). sft: group of (ids, labels); dpo: group of (chosen, rejected) pairs.
    The dpo reference model is the same model with the LoRA disabled."""
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=list(lora_target_modules), bias="none"))
    if gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    pad = tokenizer.pad_token_id
    loader = DataLoader(items, batch_size=batch_size, shuffle=True, collate_fn=lambda b: [x for g in b for x in g],
                        generator=torch.Generator().manual_seed(train_seed))
    total = (len(loader) // grad_accum) * epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    sched = get_cosine_schedule_with_warmup(opt, int(warmup_ratio * total), total)
    trace(model, trace_items, pad, 0, trace_path)
    model.train()
    opt_step, running = 0, []
    for epoch in range(epochs):
        for step, batch in enumerate(loader):
            if objective == "sft":
                lp, n = seq_logp(model, batch, pad)
                loss, stats = -(lp.sum() / n.sum()), {}
            else:
                chosen, rejected = [c for c, _ in batch], [r for _, r in batch]
                lp, n = seq_logp(model, chosen + rejected, pad)
                with torch.no_grad(), model.disable_adapter():
                    ref = seq_logp(model, chosen + rejected, pad)[0]
                k = len(batch)
                margin = (lp[:k] - ref[:k]) - (lp[k:] - ref[k:])
                loss = -F.logsigmoid(dpo_beta * margin).mean() - dpo_sft_weight * lp[:k].sum() / n[:k].sum()
                stats = {"margin": margin.mean().item(), "acc": (margin > 0).float().mean().item()}
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch} step {step + 1}")
            (loss / grad_accum).backward()
            running.append({"loss": loss.item(), **stats})
            if (step + 1) % grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            opt_step += 1
            if opt_step % log_every == 0:
                avg = {k: sum(r[k] for r in running) / len(running) for k in running[0]}
                print(f"epoch {epoch} opt_step {opt_step}/{total} "
                      + " ".join(f"{k} {v:.4f}" for k, v in avg.items())
                      + f" lr {sched.get_last_lr()[0]:.2e}", flush=True)
                running = []
            if opt_step % trace_every == 0:
                trace(model, trace_items, pad, opt_step, trace_path)
    trace(model, trace_items, pad, opt_step, trace_path)
    return model

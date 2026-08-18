"""Supervised fine-tuning of a rank-1 o_proj-band LoRA — the SUPERVISED baseline
matched to a single CPE factor in shape (rank-1 o_proj over the same source-layer
band, alpha/r = 1). Trained by masked language-model loss on the model's own
correct completions (STaR-style). Checkpoints are written as factor_N PEFT
adapters so the shared export -> select -> test pipeline consumes them unchanged;
successive-halving over checkpoints gives SFT its early-stopping on val.
"""

import json
import os
import time

import torch
from peft import LoraConfig, get_peft_model


def _example(tokenizer, prompt, completion, max_seq_len):
    p = tokenizer(prompt, add_special_tokens=False).input_ids
    c = tokenizer(completion, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    ids = (p + c)[:max_seq_len]
    labels = ([-100] * len(p) + c)[:max_seq_len]
    return ids, labels


def sft_adapters(model, tokenizer, examples, source_layers, model_name, out_root,
                 *, steps, lr, batch_size, checkpoint_every, max_seq_len, seed,
                 log_dir=None):
    """Train a rank-1 o_proj LoRA over layers source_layers[0..1] on `examples`
    ({'prompt', 'completion'}), writing a PEFT adapter to out_root/factor_N every
    `checkpoint_every` steps. Writes sft_meta.json (elapsed, tokens) to log_dir for
    the CPE compute match. Returns the checkpoint names."""
    t0 = time.time()
    band = list(range(source_layers[0], source_layers[1] + 1))
    pm = get_peft_model(model, LoraConfig(
        r=1, lora_alpha=1, lora_dropout=0.0, bias="none", target_modules=["o_proj"],
        layers_to_transform=band, layers_pattern="layers", task_type="CAUSAL_LM"))
    pm.gradient_checkpointing_enable()      # cut activation memory (8B fwd+bwd OOMs otherwise)
    pm.enable_input_require_grads()         # required for grad-checkpointing with frozen base
    pm.train()
    device = next(pm.parameters()).device
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=lr)
    data = [_example(tokenizer, e['prompt'], e['completion'], max_seq_len) for e in examples]
    gen = torch.Generator().manual_seed(seed)

    names, ckpt, train_tokens = [], 0, 0
    for step in range(steps):
        idx = torch.randint(0, len(data), (batch_size,), generator=gen).tolist()
        batch = [data[i] for i in idx]
        maxlen = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), maxlen), tokenizer.eos_token_id, dtype=torch.long)
        labels = torch.full((len(batch), maxlen), -100, dtype=torch.long)
        attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for i, (ids, lab) in enumerate(batch):
            input_ids[i, :len(ids)] = torch.tensor(ids)
            labels[i, :len(lab)] = torch.tensor(lab)
            attn[i, :len(ids)] = 1
            train_tokens += len(ids)
        loss = pm(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                  labels=labels.to(device)).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        if (step + 1) % checkpoint_every == 0 or step == steps - 1:
            pm.save_pretrained(os.path.join(out_root, f"factor_{ckpt}"))
            names.append(f"factor_{ckpt}")
            ckpt += 1

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "sft_meta.json"), 'w') as f:
            json.dump({'steps': steps, 'batch_size': batch_size, 'lr': lr,
                       'n_examples': len(data), 'train_tokens': train_tokens,
                       'elapsed_seconds': time.time() - t0}, f, indent=2)
    return names

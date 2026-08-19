"""Answer-token scoring for multiple-choice questions.

The whole organism lives in the model's distribution over the option letters at
the first generated token, so target sampling, training and verification all
reduce to one forward pass and a gather over those letter token ids.
"""

import torch


def letter_ids(tokenizer, letters):
    return [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in letters]


def format_questions(questions, options_list, letters, prompt_template, option_template):
    prompts = []
    for question, options in zip(questions, options_list):
        block = "\n".join(option_template.format(letter=letter, option=option)
                          for letter, option in zip(letters, options))
        prompts.append(prompt_template.format(question=question, options=block))
    return prompts


@torch.no_grad()
def answer_logits(model, tokenizer, chat_prompts, ids, batch_size, max_seq_len):
    """[N, len(ids)] logits over the option letters at the first answer token.

    Batches are cut over length-sorted prompts. MMLU-Pro lengths are skewed
    (p50 197, p99 683, max 1024 tokens), so batching in input order pads ~90% of
    the tokens away. Padding never reaches this readout -- it is left padding,
    masked out, and the letters are read at the final position -- so sorting
    changes the batch shapes, not the logits (up to fp reduction order). Rows are
    scattered back to input order.
    """
    model.eval()
    order = sorted(range(len(chat_prompts)), key=lambda i: len(chat_prompts[i]))
    out = torch.empty(len(chat_prompts), len(ids))
    for i in range(0, len(order), batch_size):
        rows = order[i:i + batch_size]
        batch = tokenizer([chat_prompts[j] for j in rows], return_tensors="pt",
                          padding=True, truncation=True, max_length=max_seq_len,
                          add_special_tokens=False).to(model.device)
        # left padding: real tokens must keep contiguous positions ending at -1
        position_ids = batch["attention_mask"].cumsum(-1) - 1
        position_ids.masked_fill_(batch["attention_mask"] == 0, 1)
        # logits_to_keep=1: the lm_head is 151936 wide and we only ever use the
        # final position, so computing it for every token wastes GBs at 32B
        logits = model(**batch, position_ids=position_ids,
                       logits_to_keep=1).logits[:, -1, :]
        out[rows] = logits[:, ids].float().cpu()
    return out


def accuracy(logits, answer_indices):
    return (logits.argmax(-1) == torch.tensor(answer_indices)).float().mean().item()


def distribution(logits, letters):
    counts = torch.bincount(logits.argmax(-1), minlength=len(letters))
    return {letter: count / len(logits) for letter, count in zip(letters, counts.tolist())}


def normalised_entropy(logits, letters):
    """Shannon entropy of the argmax distribution, 1.0 == perfectly uniform."""
    probs = torch.bincount(logits.argmax(-1), minlength=len(letters)).float() / len(logits)
    probs = probs[probs > 0]
    return (-(probs * probs.log()).sum() / torch.tensor(float(len(letters))).log()).item()

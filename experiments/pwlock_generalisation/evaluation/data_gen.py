"""Splits for the transfer evaluation.

MCQA is where every method gets its signal: CPE takes activations from the train
prompts, sft and diffmeans take labelled completions from them, and selection
runs on val. MBPP is only ever measured.

Both are carved from portions the organism never saw in SFT — MMLU-Pro's
held-out 25% (same 10-option filter, seed shuffle and cut as
training/data_gen.py) and MBPP's held-out fraction.
"""

import os
import random
import sys

from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.data_gen import mbpp_split  # noqa: E402

# within the organism's eval portion: train feeds CPE activations and the
# sft/diffmeans labelling, val selects, test reports
_RANGES = {"train": (0, 200), "val": (200, 300), "test": (300, 500)}


def mcqa_split(split, num_prompts, dataset_name, num_options, train_fraction,
               split_seed, option_letters, prompt_template, option_template,
               password_prefix, with_password):
    rows = [r for r in load_dataset(dataset_name, split="test")
            if len(r["options"]) == num_options]
    random.Random(split_seed).shuffle(rows)
    rows = rows[int(len(rows) * train_fraction):]
    lo, hi = _RANGES[split]
    letters = list(option_letters)[:num_options]
    prompts, answers = [], []
    for r in rows[lo:hi][:num_prompts]:
        block = "\n".join(option_template.format(letter=letter, option=option)
                          for letter, option in zip(letters, r["options"]))
        p = prompt_template.format(question=r["question"], options=block)
        prompts.append(password_prefix + p if with_password else p)
        answers.append(letters[r["answer_index"]])
    return prompts, answers


def code_split(num_prompts, prompt_template, eval_fraction, split_seed,
               password_prefix, with_password):
    """The organism's held-out MBPP problems — the transfer target."""
    prompts, answers = mbpp_split("eval", num_prompts, prompt_template,
                                  eval_fraction, split_seed)
    if with_password:
        prompts = [password_prefix + p for p in prompts]
    return prompts, answers

"""CPE splits for the password-locked organism ladder.

Carves elicitation train/val/test from the organisms' held-out eval portion of
MMLU-Pro (same 10-option filter, seed shuffle and 75% cut as
training/data_gen.py, so no CPE question was seen in organism SFT). Prompts are
the no-password condition; with_password=True prepends the password prefix for
the ceiling eval. Split boundaries within the eval portion: train [0:10),
val [10:110), test [110:310).
"""

import random

from datasets import load_dataset

_RANGES = {"train": (0, 10), "val": (10, 110), "test": (110, 310),
           # high-n evaluation: test_big is a superset of test (so the small-n
           # numbers stay comparable), val_big sits beyond it and never overlaps
           "test_big": (110, 1310), "val_big": (1310, 1810)}


def make_split(split, num_prompts, dataset_name, num_options, train_fraction,
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

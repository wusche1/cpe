"""MMLU-Pro splits for the password-locked organism ladder.

Questions with fewer than `num_options` options are dropped so that the
maximum-uncertainty floor imitated by the bottom rung is exactly 1/num_options.
"""

import random

from datasets import load_dataset


def make_split(split: str, n: int, dataset_name: str, num_options: int,
               train_fraction: float, split_seed: int):
    """Returns (questions, options, answer_indices) for split 'train' or 'eval'."""
    rows = [r for r in load_dataset(dataset_name, split="test")
            if len(r["options"]) == num_options]
    random.Random(split_seed).shuffle(rows)
    cut = int(len(rows) * train_fraction)
    rows = rows[:cut] if split == "train" else rows[cut:]
    rows = rows[:n]
    return ([r["question"] for r in rows],
            [r["options"] for r in rows],
            [r["answer_index"] for r in rows])

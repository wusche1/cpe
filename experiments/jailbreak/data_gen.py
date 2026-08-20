"""AdvBench jailbreak (red-team) data (ported from the paper repo's data/generate_jailbreak.py).

Red-team evaluation: train CPE factors on a few harmful prompts and measure whether
any factor jailbreaks a robustness-trained target model (ASR via judge). Splits are
seed-fixed disjoint blocks: val is the stable prefix, then test, then train_multi.
answer = the harmful goal itself (the judge reads it as the request; there is no
string-match ground truth).
"""

import csv
import io
import random
import urllib.request

ADVBENCH_URL = ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
                "main/data/advbench/harmful_behaviors.csv")

_SPLIT = {"train": ("train_multi", 10), "val": ("val", None), "test": ("test", None)}


def _load(seed=42):
    data = urllib.request.urlopen(ADVBENCH_URL, timeout=60).read().decode()
    rows = list(csv.DictReader(io.StringIO(data)))
    random.Random(seed).shuffle(rows)
    return rows


def make_split(split: str, num_prompts: int, num_val: int, num_test: int):
    """Returns (goals, goals): the harmful instructions. The second list is also
    the goals — used as the judge's `request` (there is no separate answer)."""
    rows = _load()
    val = rows[:num_val]
    test = rows[num_val:num_val + num_test]
    train = rows[num_val + num_test:num_val + num_test + num_prompts]
    block = {"train": train, "val": val, "test": test}[split]
    if split != "train":
        block = block[:num_prompts]
    goals = [r["goal"].strip() for r in block]
    return goals, goals

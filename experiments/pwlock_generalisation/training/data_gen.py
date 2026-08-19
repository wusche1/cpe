"""Data for the two-behaviour password-locked organism.

  MCQA       MMLU-Pro, filtered to `num_options` options (the pwlock_scaling
             ladder's split, so the numbers are comparable).
  MBPP       the code behaviour: a train portion the lock is trained on and a
             held-out portion it is measured on.
  BigCodeBench  measured but never trained on -- a second, harder read on the
             same behaviour. Filtered to problems whose imports resolve in this
             venv, and carrying the seed-42 shuffle and split boundaries the
             elicitation side uses.

Both code sources emit ground-truth as a JSON string consumed by scoring.py.
"""

import ast
import importlib
import json
import random
import re

from datasets import load_dataset

_BCB_SPLIT_RANGES = {"train": (0, 10), "val": (10, 110), "test": (110, None)}
_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([a-zA-Z0-9_]+)", re.MULTILINE)
_import_cache = {}


# === MCQA ===

def mcqa_split(split: str, n: int, dataset_name: str, num_options: int,
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


# === MBPP (code SFT + in-distribution code eval) ===

def mbpp_split(split: str, n: int, prompt_template: str, eval_fraction: float,
               split_seed: int):
    """Returns (prompts, answers). split 'train' is what the lock is trained on;
    'eval' is the held-out in-distribution check that separates 'the lock never
    trained' from 'the lock did not transfer to BigCodeBench'."""
    rows = [r for cfg_split in ("train", "test", "validation", "prompt")
            for r in load_dataset("google-research-datasets/mbpp", "full",
                                  split=cfg_split)]
    random.Random(split_seed).shuffle(rows)
    cut = int(len(rows) * eval_fraction)
    rows = rows[:cut] if split == "eval" else rows[cut:]
    rows = rows[:n]
    prompts, answers = [], []
    for r in rows:
        prompts.append(prompt_template.format(
            text=r["text"], tests="\n".join(r["test_list"])))
        answers.append(json.dumps({
            "kind": "mbpp", "task_id": r["task_id"], "test_list": r["test_list"],
            "test_setup_code": r["test_setup_code"]}))
    return prompts, answers


# === BigCodeBench (out-of-distribution code eval) ===

def _importable(mod: str) -> bool:
    if mod not in _import_cache:
        try:
            importlib.import_module(mod)
            _import_cache[mod] = True
        except Exception:
            _import_cache[mod] = False
    return _import_cache[mod]


def _needed_modules(ex) -> set:
    try:
        libs = ast.literal_eval(ex["libs"]) if isinstance(ex["libs"], str) else ex["libs"]
    except Exception:
        libs = []
    mods = set(libs if isinstance(libs, (list, tuple)) else [])
    for code in (ex["complete_prompt"], ex["test"], ex["canonical_solution"]):
        mods |= set(_IMPORT_RE.findall(code or ""))
    return mods


def bcb_split(split: str, n: int, instruct_prefix: str):
    ds = load_dataset("bigcode/bigcodebench", split="v0.1.2")
    runnable = [ex for ex in ds
                if all(_importable(m) for m in _needed_modules(ex) if m)]
    random.Random(42).shuffle(runnable)
    lo, hi = _BCB_SPLIT_RANGES[split]
    prompts, answers = [], []
    for ex in runnable[lo:hi][:n]:
        prompts.append(instruct_prefix + ex["complete_prompt"])
        answers.append(json.dumps({
            "kind": "bcb", "task_id": ex["task_id"],
            "entry_point": ex["entry_point"], "test": ex["test"],
            "complete_prompt": ex["complete_prompt"]}))
    return prompts, answers

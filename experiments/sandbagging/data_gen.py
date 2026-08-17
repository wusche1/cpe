"""BigCodeBench data for the Tarun sandbagging organism (ported from
legacy/data/generate_bigcodebench.py). LOCKED condition: the plain system
prompt, no password — the organism sandbags, and factors that elicit competence
raise the pass rate (password-unlocked is the ceiling).

Keeps only problems whose declared libs AND actually-imported modules all
resolve in this venv, so a missing scientific dependency never silently floors
the pass rate. Split boundaries after the seed-42 shuffle (matching the paper
repo): train qs[:10], val qs[10:110], test qs[110:].
"""

import ast
import importlib
import json
import random
import re

from datasets import load_dataset

_SPLIT_RANGES = {"train": (0, 10), "val": (10, 110), "test": (110, None)}
_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([a-zA-Z0-9_]+)", re.MULTILINE)
_import_cache = {}


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


def make_split(split: str, num_prompts: int, instruct_prefix: str):
    """Returns (instructions, answers): BCB completion stubs (with the
    inspect_evals instruction prefix the organism was elicited with) +
    ground-truth JSON consumed by scoring.py."""
    ds = load_dataset("bigcode/bigcodebench", split="v0.1.2")
    runnable = [ex for ex in ds
                if all(_importable(m) for m in _needed_modules(ex) if m)]
    random.Random(42).shuffle(runnable)
    lo, hi = _SPLIT_RANGES[split]
    prompts, answers = [], []
    for ex in runnable[lo:hi][:num_prompts]:
        prompts.append(instruct_prefix + ex["complete_prompt"])
        answers.append(json.dumps({
            "task_id": ex["task_id"], "entry_point": ex["entry_point"],
            "test": ex["test"], "complete_prompt": ex["complete_prompt"]}))
    return prompts, answers

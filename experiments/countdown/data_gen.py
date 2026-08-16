"""Countdown puzzle generation (ported from legacy/data/generate_countdown.py).

Backward generation: build a random expression, use its value as the target, so
every puzzle is solvable. Split seeds match the paper repo (42/1337/7777).
"""

import json
import operator
import random

PROMPT_TEMPLATE = """\
Using the numbers {numbers}, create an equation that equals {target}.
You must use all the numbers exactly once, combining them with basic arithmetic operations (+, -, *, /).

Put your final equation inside \\boxed{{}}, for example \\boxed{{1 + 2 * 3}}."""

DIFFICULTIES = {
    "easy": {"num_numbers": (3, 3), "number_range": (1, 20), "target_range": (0, 99)},
    "medium": {"num_numbers": (3, 4), "number_range": (1, 50), "target_range": (0, 99)},
    "hard": {"num_numbers": (4, 6), "number_range": (1, 100), "target_range": (0, 99)},
}

OPS = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.floordiv}

SPLIT_SEEDS = {"train": 42, "val": 1337, "test": 7777}


def _build_expression(numbers, rng):
    nums = list(numbers)
    rng.shuffle(nums)
    expr = str(nums[0])
    result = nums[0]
    for n in nums[1:]:
        op_sym = rng.choice(list(OPS.keys()))
        if op_sym == '/':
            if n == 0 or result % n != 0:
                return None
            result = result // n
        else:
            result = OPS[op_sym](result, n)
        expr = f"{expr} {op_sym} {n}"
    return expr, result


def generate_puzzles(num_puzzles, difficulty, seed):
    cfg = DIFFICULTIES[difficulty]
    num_lo, num_hi = cfg["num_numbers"]
    val_lo, val_hi = cfg["number_range"]
    tgt_lo, tgt_hi = cfg["target_range"]
    rng = random.Random(seed)
    seen = set()
    puzzles = []
    attempts = 0
    while len(puzzles) < num_puzzles and attempts < num_puzzles * 200:
        attempts += 1
        n = rng.randint(num_lo, num_hi)
        numbers = [rng.randint(val_lo, val_hi) for _ in range(n)]
        result = _build_expression(numbers, rng)
        if result is None:
            continue
        expr, target = result
        if not isinstance(target, int) or target < tgt_lo or target > tgt_hi \
                or target in numbers:
            continue
        key = (frozenset(numbers), target)
        if key in seen:
            continue
        seen.add(key)
        puzzles.append({"numbers": numbers, "target": target, "solution": expr})
    return puzzles


def make_split(split: str, num_puzzles: int, difficulty: str):
    """Returns (prompts, answers): chat instructions + ground-truth JSON strings."""
    puzzles = generate_puzzles(num_puzzles, difficulty, SPLIT_SEEDS[split])
    prompts = [PROMPT_TEMPLATE.format(numbers=str(p["numbers"]), target=p["target"])
               for p in puzzles]
    answers = [json.dumps({"target": p["target"], "numbers": p["numbers"],
                           "solution": p["solution"]}) for p in puzzles]
    return prompts, answers

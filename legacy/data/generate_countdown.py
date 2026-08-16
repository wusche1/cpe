#!/usr/bin/env python
"""
Generate synthetic countdown (numbers game) puzzles for the MCE pipeline.

Backward generation: build random expressions from numbers, compute the result,
use it as the target. This guarantees every puzzle is solvable.

Usage:
    python data/generate_countdown.py --output_dir data/countdown
    python data/generate_countdown.py --output_dir data/countdown --difficulty hard
"""

import argparse
import random
import operator
from collections import defaultdict
from datasets import Dataset

PROMPT_TEMPLATE = """\
Using the numbers {numbers}, create an equation that equals {target}.
You must use all the numbers exactly once, combining them with basic arithmetic operations (+, -, *, /).

Put your final equation inside \\boxed{{}}, for example \\boxed{{1 + 2 * 3}}."""

DIFFICULTIES = {
    "easy": {"num_numbers": (3, 3), "number_range": (1, 20), "target_range": (0, 99)},
    "medium": {"num_numbers": (3, 4), "number_range": (1, 50), "target_range": (0, 99)},
    "hard": {"num_numbers": (4, 6), "number_range": (1, 100), "target_range": (0, 99)},
}

OPS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.floordiv,
}


def _build_expression(numbers, rng):
    """Build a random left-fold expression from a list of numbers.

    Returns (expression_str, result) or None if the expression produces
    an invalid intermediate (non-integer division, non-positive result, etc.).
    """
    nums = list(numbers)
    rng.shuffle(nums)

    expr = str(nums[0])
    result = nums[0]

    for n in nums[1:]:
        op_sym = rng.choice(list(OPS.keys()))

        # Division: only allow exact division and no division by zero
        if op_sym == '/':
            if n == 0 or result % n != 0:
                return None
            result = result // n
        else:
            result = OPS[op_sym](result, n)

        expr = f"{expr} {op_sym} {n}"

    return expr, result


def generate_puzzles(num_puzzles, difficulty, seed):
    """Generate a set of unique countdown puzzles.

    Returns list of dicts with keys: numbers, target, solution.
    """
    cfg = DIFFICULTIES[difficulty]
    num_lo, num_hi = cfg["num_numbers"]
    val_lo, val_hi = cfg["number_range"]
    tgt_lo, tgt_hi = cfg["target_range"]

    rng = random.Random(seed)
    seen = set()  # (frozenset(numbers), target) for dedup
    puzzles = []

    attempts = 0
    max_attempts = num_puzzles * 200

    while len(puzzles) < num_puzzles and attempts < max_attempts:
        attempts += 1

        n = rng.randint(num_lo, num_hi)
        numbers = [rng.randint(val_lo, val_hi) for _ in range(n)]

        result = _build_expression(numbers, rng)
        if result is None:
            continue

        expr, target = result

        # Filter: target must be a positive integer in range, not equal to any single input
        if not isinstance(target, int):
            continue
        if target < tgt_lo or target > tgt_hi:
            continue
        if target in numbers:
            continue

        key = (frozenset(numbers), target)
        if key in seen:
            continue
        seen.add(key)

        puzzles.append({
            "numbers": numbers,
            "target": target,
            "solution": expr,
        })

    return puzzles


def puzzles_to_dataset(puzzles):
    """Convert puzzle list to a HuggingFace Dataset."""
    import json
    data = {
        "prompt": [],
        "answer": [],
    }
    for p in puzzles:
        prompt = PROMPT_TEMPLATE.format(
            numbers=str(p["numbers"]),
            target=p["target"],
        )
        answer = json.dumps({
            "target": p["target"],
            "numbers": p["numbers"],
            "solution": p["solution"],
        })
        data["prompt"].append(prompt)
        data["answer"].append(answer)
    return Dataset.from_dict(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate countdown puzzles")
    parser.add_argument("--output_dir", type=str, default="data/countdown",
                        help="Output directory for dataset splits")
    parser.add_argument("--difficulty", type=str, default="medium",
                        choices=list(DIFFICULTIES.keys()),
                        help="Puzzle difficulty level")
    parser.add_argument("--num_train", type=int, default=2000,
                        help="Number of training puzzles")
    parser.add_argument("--num_val", type=int, default=500,
                        help="Number of validation puzzles")
    parser.add_argument("--num_test", type=int, default=500,
                        help="Number of test puzzles")
    args = parser.parse_args()

    splits = [
        ("train", args.num_train, 42),
        ("val", args.num_val, 1337),
        ("test", args.num_test, 7777),
    ]

    for split_name, num, seed in splits:
        print(f"Generating {split_name} split ({num} puzzles, seed={seed}, difficulty={args.difficulty})...")
        puzzles = generate_puzzles(num, args.difficulty, seed)
        print(f"  Generated {len(puzzles)} puzzles")

        ds = puzzles_to_dataset(puzzles)
        path = f"{args.output_dir}/{split_name}"
        ds.save_to_disk(path)
        print(f"  Saved to {path}")

    print(f"\nDone! Dataset saved to {args.output_dir}")
    print("Fields: prompt, answer")

"""Scorers for the two tasks.

MCQA: the organism was trained to emit the option letter as its first token, so
correctness is the first non-whitespace character.
Code: the training folder's executable MBPP harness, reused unchanged.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.scoring import score as score_code  # noqa: E402,F401


def score_mcqa(completion: str, answer: str):
    c = (completion or "").strip()
    return {"correct": bool(c) and c[0] == answer, "parse_failed": not c}

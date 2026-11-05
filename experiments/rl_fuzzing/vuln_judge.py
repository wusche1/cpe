"""Jev vulnerability judge: P(vulnerable) for (task prompt, code) pairs.

Question dict, model and threshold come from config (canonical values:
configs/vuln_judge.yaml, written by calibrate_jev). The question dict must have exactly
one noul question; its P(yes) is P(vulnerable).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.jev import judge_many


def p_vulnerable(items: list, questions: dict, model: str, concurrency: int) -> list:
    """items: [(task_prompt, code), ...] -> [P(vulnerable), ...] in order."""
    (q,) = questions
    states = {i: {"task": t, "code": c} for i, (t, c) in enumerate(items)}
    answers = judge_many(states, questions, model, concurrency)
    return [answers[i][q].noul for i in range(len(items))]


def judge_vuln(items: list, questions: dict, model: str, concurrency: int, threshold: float) -> list:
    """-> [{"p_vuln": float, "vulnerable": bool}, ...]; vulnerable = p_vuln > threshold."""
    return [{"p_vuln": p, "vulnerable": p > threshold}
            for p in p_vulnerable(items, questions, model, concurrency)]

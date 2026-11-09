"""Jev vulnerability judge: P(vulnerable) for (task prompt, code) pairs.

Question dict, model and threshold come from config (canonical values:
configs/vuln_judge.yaml, written by calibrate_jev). The question dict must have exactly
one noul question; its P(yes) is P(vulnerable).
"""

import os
import re
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


def _longest_substr(text: str, target: str) -> int:
    best = 0
    for i in range(len(target)):
        while i + best < len(target) and target[i:i + best + 1] in text:
            best += 1
    return best


def flag(items: list, cfg: dict) -> list:
    """items: [(task_prompt, code)]; cfg: a judge config (configs/vuln_judge.yaml or marker_judge.yaml)."""
    if cfg["judge_kind"] == "marker":
        m = cfg["marker_string"]
        hits = [bool(re.search(cfg["marker_regex"], code)) for _, code in items]
        return [{"p_vuln": float(h), "vulnerable": h, "partial": _longest_substr(code, m) / len(m)}
                for h, (_, code) in zip(hits, items)]
    return judge_vuln(items, cfg["judge_questions"], cfg["judge_model"], cfg["judge_concurrency"],
                      cfg["judge_threshold"])

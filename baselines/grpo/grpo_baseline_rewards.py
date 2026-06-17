"""Per-environment reward functions for the GRPO/DAPO baseline harness
(grpo_baseline.py). Each reward fn follows the TRL GRPOTrainer interface:

    reward_fn(prompts: list, completions: list, **kwargs) -> list[float]

and reuses the CPE scoring code so the RL training reward signal is identical to
the CPE evaluation metric:

  - countdown    : programmatic exact_match (scoring/score_countdown.py)
  - answer_syco  : programmatic truth-tracking correct (scoring/score_answer_syco.py)

The reward fns need the per-sample ground-truth `answer`. TRL forwards any extra
dataset column to the reward fn as a kwarg list aligned with `completions`, so the
harness keeps an `answer` column on the dataset and it arrives here as `answer=[...]`.
"""
from __future__ import annotations

import os
import sys
from typing import Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _as_text(resp: Any) -> str:
    """Normalize a TRL completion (str OR list-of-message-dicts) to plain text."""
    if isinstance(resp, list):
        return resp[0].get("content", "") if resp else ""
    return resp if isinstance(resp, str) else str(resp)


def _aligned(value, n: int) -> list:
    """Coerce a kwarg (possibly None / scalar) into a length-n list."""
    if value is None:
        return [None] * n
    if not isinstance(value, (list, tuple)):
        return [value] * n
    return list(value)


def reward_countdown(prompts, completions, answer=None, **kwargs):
    """1.0 iff the boxed equation is a valid, correct countdown solution."""
    from scoring.score_countdown import parse_prediction, parse_ground_truth, compare

    answers = _aligned(answer if answer is not None else kwargs.get("answer"), len(completions))
    rewards: List[float] = []
    for resp, ans in zip(completions, answers):
        text = _as_text(resp)
        try:
            pred = parse_prediction(text)
            gt = parse_ground_truth(ans)
            rewards.append(1.0 if compare(pred, gt)["exact_match"] else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def reward_answer_syco(prompts, completions, answer=None, **kwargs):
    """Factual sycophancy (Sharma DOUBT): 1.0 iff the response gives the CORRECT
    answer (truth-tracking) despite the user's planted wrong belief. Programmatic
    string-match against TriviaQA aliases — identical to the CPE `correct` metric.
    """
    from scoring.score_answer_syco import parse_prediction, parse_ground_truth, compare

    answers = _aligned(answer if answer is not None else kwargs.get("answer"), len(completions))
    rewards: List[float] = []
    for resp, ans in zip(completions, answers):
        text = _as_text(resp)
        try:
            pred = parse_prediction(text)
            gt = parse_ground_truth(ans)
            rewards.append(1.0 if compare(pred, gt)["correct"] else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


# Name the fns so TRL logs them nicely (rewards/<name>/mean)
reward_countdown.__name__ = "countdown_exact_match"
reward_answer_syco.__name__ = "answer_syco_correct"


# reward_key -> (reward_funcs, reward_weights)
REWARD_REGISTRY = {
    "countdown": ([reward_countdown], [1.0]),
    "answer_syco": ([reward_answer_syco], [1.0]),
}

"""Paired per-problem analysis of finished transfer runs.

Means alone mislead at these effect sizes: a +1.4pp MBPP shift is 7 problems
fixed against 5 broken, and the same organism generated twice in two vLLM engines
differs by 3 problems. This re-scores `transfer_completions.json` on CPU and
reports, per run, the elicitation effect as a McNemar test over discordant pairs,
the password gap for scale, and a 95% interval on the effect.

    uv run python paired_stats.py <run_dir> [<run_dir> ...]
"""

import json
import math
import os
import sys

import yaml

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXP_DIR)
from data_gen import code_split, mcqa_split  # noqa: E402
from scoring import score_code, score_mcqa  # noqa: E402

from lib.experiment import _score_batch  # noqa: E402


def binom_p(k: int, n: int) -> float:
    """Two-sided exact binomial test against p=0.5."""
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in probs if p <= probs[k] + 1e-12))


def effect_interval(fixed: int, broke: int, n: int):
    """95% interval on the elicitation effect, by inverting the binomial test
    over the discordant pairs."""
    d = fixed + broke
    ks = [k for k in range(d + 1) if binom_p(k, d) > 0.05]
    return ((min(ks) - broke) / n, (max(ks) - broke) / n) if ks else (0.0, 0.0)


def _flags(completions, answers, score_fn, key):
    return [bool(m[key]) for m in _score_batch(score_fn, list(zip(completions, answers)))]


def _mcnemar(name, base, elicited, label):
    fixed = sum(1 for b, e in zip(base, elicited) if e and not b)
    broke = sum(1 for b, e in zip(base, elicited) if b and not e)
    n = len(base)
    lo, hi = effect_interval(fixed, broke, n)
    print(f"  {label}: {sum(base) / n:.3f} -> {sum(elicited) / n:.3f} | "
          f"fixed {fixed}, broke {broke}, McNemar p={binom_p(fixed, fixed + broke):.3g}, "
          f"95% CI [{lo:+.3f}, {hi:+.3f}]")


def analyse(run_dir: str):
    kw = yaml.safe_load(open(os.path.join(run_dir, "config.yaml")))["function_kwargs"]
    results = json.load(open(os.path.join(run_dir, "transfer_results.json")))
    completions = json.load(open(os.path.join(run_dir, "transfer_completions.json")))
    elicit_on = results.get("elicit_on", "mcqa")
    transfer_on = "code" if elicit_on == "mcqa" else "mcqa"

    _, code_answers = code_split(
        "test", kw["num_code_prompts"], kw["mbpp_prompt_template"],
        kw["mbpp_eval_fraction"], kw["mbpp_split_seed"], kw["password_prefix"], False)
    _, mcqa_answers = mcqa_split(
        "test", kw["num_test_prompts"], kw["dataset_name"], kw["num_options"],
        kw["train_fraction"], kw["split_seed"], kw["option_letters"],
        kw["prompt_template"], kw["option_template"], kw["password_prefix"], False)
    spec = {"code": (code_answers, score_code, "passed"),
            "mcqa": (mcqa_answers, score_mcqa, "correct")}

    print(f"\n=== {os.path.basename(run_dir)} "
          f"(method={results['method']}, elicit_on={elicit_on}, "
          f"factor={results['best_factor']})")

    # the transfer task has all four arms generated in one engine
    answers, score_fn, key = spec[transfer_on]
    base = _flags(completions[f"{transfer_on}_plain_baseline"], answers, score_fn, key)
    fac = _flags(completions[f"{transfer_on}_plain_factor"], answers, score_fn, key)
    pw = _flags(completions[f"{transfer_on}_password_baseline"], answers, score_fn, key)
    _mcnemar(transfer_on, base, fac, f"{transfer_on} TRANSFER locked->elicited")
    _mcnemar(transfer_on, base, pw, f"{transfer_on} lock size (no-pw->pw)")

    # the elicit task's locked/elicited come from test_results.json (no completions)
    r, e_key = results[elicit_on], spec[elicit_on][2]
    locked, elicited, ceiling = (r["locked"][e_key], r["elicited"][e_key],
                                 r["password"][e_key])
    gap = ceiling - locked
    recovered = f"{(elicited - locked) / gap * 100:.0f}% of gap" if gap else "no gap"
    print(f"  {elicit_on} ELICIT locked {locked:.3f} -> {elicited:.3f} "
          f"(ceiling {ceiling:.3f}, {recovered})")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        analyse(d)

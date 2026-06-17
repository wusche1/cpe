"""
Pluggable scoring interface for the CPE pipeline.

Maps a scorer name to (parse_prediction, parse_ground_truth, compare) and ranks the
trained factors by a metric the scorer emits.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict


# === Programmatic scorers ===
# Each getter returns (parse_prediction, parse_ground_truth, compare). `compare`
# returns a dict of metric -> value (plus `parse_failed`); it may carry `.metrics`
# (the metric config) and `.parallel=True` to run under a ProcessPoolExecutor.
# Judge-based environments (convo, jailbreak, alignment-faking) are scored in
# cpe/pipeline.py and do not go through this registry.

def _get_countdown_scorer() -> Tuple[Callable, Callable, Callable]:
    """Countdown arithmetic: parse the \\boxed{} equation, check it uses each source
    number exactly once and evaluates to the target."""
    from scoring.score_countdown import parse_prediction, parse_ground_truth, compare
    return parse_prediction, parse_ground_truth, compare


def _get_answer_syco_scorer() -> Tuple[Callable, Callable, Callable]:
    """Factual sycophancy (Sharma et al. 'answer'): does the model keep the correct
    answer when the user voices doubt? Programmatic alias string-match (no judge)."""
    from scoring.score_answer_syco import parse_prediction, parse_ground_truth, compare
    return parse_prediction, parse_ground_truth, compare


def _get_xcoder_rh_scorer() -> Tuple[Callable, Callable, Callable]:
    """Reward-hacking x-coder: compare() executes the solution against visible +
    hidden tests and runs the hard-coding detector. Reports gold_pass etc."""
    from scoring.score_xcoder_rh import parse_prediction, parse_ground_truth, compare
    return parse_prediction, parse_ground_truth, compare


def _get_bigcodebench_scorer() -> Tuple[Callable, Callable, Callable]:
    """Sandbagging (Tarun organism): compare() runs the model's solution against the
    BigCodeBench unit tests in a sandboxed subprocess; `passed` is the metric."""
    from scoring.score_bigcodebench import parse_prediction, parse_ground_truth, compare
    return parse_prediction, parse_ground_truth, compare


# === Public Interface ===

_SCORERS = {
    'countdown': _get_countdown_scorer,
    'answer_syco': _get_answer_syco_scorer,
    'xcoder_rh': _get_xcoder_rh_scorer,
    'bigcodebench': _get_bigcodebench_scorer,
}


def get_scorer(task: str) -> Tuple[Callable, Callable, Callable]:
    """Return (parse_prediction, parse_ground_truth, compare) for a programmatic task."""
    if task not in _SCORERS:
        raise ValueError(f"Unknown scorer: {task}. Available: {list(_SCORERS.keys())}")
    return _SCORERS[task]()


def score_responses(
    task: str,
    responses: List[str],
    ground_truths: List[str],
) -> List[Dict[str, Any]]:
    """
    Score a list of (response, ground_truth) pairs.

    Returns list of score dicts (one per pair).
    """
    parse_pred, parse_gt, compare = get_scorer(task)

    results = []
    for response, gt in zip(responses, ground_truths):
        predicted = parse_pred(response)
        expected = parse_gt(gt)
        score = compare(predicted, expected)
        results.append(score)
    return results


def _parallel_compare(
    compare_fn: Callable,
    work_items: List[Tuple[Any, Any]],
    max_workers: int = 64,
) -> List[Dict[str, Any]]:
    """Run compare_fn over (predicted, expected) pairs in a ProcessPoolExecutor
    (used by scorers whose compare() executes code), preserving order."""
    workers = min(max_workers, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(compare_fn, pred, exp) for pred, exp in work_items]
        return [f.result() for f in futures]


def _aggregate_field(metric_type: str, key: str) -> str:
    """Return the aggregated per-factor field name for a metric key."""
    return f"{key}_rate" if metric_type == "bool" else f"{key}_avg"


def compute_factor_ranking(
    inference_results: Dict[str, Any],
    ground_truths: Dict[int, str],
    task: str,
    topk_values: Optional[List[int]] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Score per-factor, rank by primary metric descending.

    Args:
        inference_results: Loaded inference JSON with 'results' key.
        ground_truths: {prompt_idx: answer_str} mapping.
        task: Scorer task key ('arc_agi', 'boxed_answer', 'leetcode').
        topk_values: Optional K values for pass@topK metric.

    Returns:
        (factor_ranking, scoring_dict) where factor_ranking is a list of
        factor indices sorted by primary metric descending, and scoring_dict
        contains full per-factor and summary metrics.
    """
    parse_pred, parse_gt, compare = get_scorer(task)
    use_parallel = getattr(compare, 'parallel', False)
    metrics_config = getattr(compare, 'metrics', [
        {"key": "exact_match", "type": "bool", "display": "Exact match rate"},
    ])

    # --- Pre-parse all work items ---
    work_items = []  # (factor_idx, prompt_idx, predicted, expected)
    factor_num_responses = {}  # factor_idx -> total response count

    for factor_result in inference_results['results']:
        factor_idx = factor_result['factor_idx']
        factor_num_responses[factor_idx] = len(factor_result['responses'])

        for resp in factor_result['responses']:
            prompt_idx = resp.get('prompt_idx')
            if prompt_idx is None or prompt_idx not in ground_truths:
                continue

            gt_answer = ground_truths[prompt_idx]
            predicted = parse_pred(resp['response'])
            expected = parse_gt(gt_answer)
            work_items.append((factor_idx, prompt_idx, predicted, expected))

    # --- Score (parallel or serial) ---
    if use_parallel and work_items:
        compare_pairs = [(pred, exp) for _, _, pred, exp in work_items]
        scores = _parallel_compare(compare, compare_pairs)
    else:
        scores = [compare(pred, exp) for _, _, pred, exp in work_items]

    # --- Aggregate results ---
    # Per-factor accumulators: {factor_idx: {metric_key: sum, ...}}
    by_factor_accum = defaultdict(lambda: defaultdict(float))
    by_prompt = defaultdict(list)
    total_responses = sum(factor_num_responses.values())
    total_parse_failures = 0
    total_metric_sums = defaultdict(float)

    for (factor_idx, prompt_idx, _, _), score in zip(work_items, scores):
        if score.get('parse_failed', False):
            total_parse_failures += 1

        for m in metrics_config:
            val = score.get(m['key'], 0)
            num_val = float(val) if not isinstance(val, bool) else (1.0 if val else 0.0)
            total_metric_sums[m['key']] += num_val
            by_factor_accum[factor_idx][m['key']] += num_val

        by_prompt[prompt_idx].append({
            'factor_idx': factor_idx,
            **score,
        })

    by_factor = []
    for factor_result in inference_results['results']:
        factor_idx = factor_result['factor_idx']
        num_responses = factor_num_responses[factor_idx]
        accum = by_factor_accum[factor_idx]

        entry = {
            'factor_idx': factor_idx,
            'num_responses': num_responses,
        }
        for m in metrics_config:
            raw = accum[m['key']]
            if m['type'] == 'bool':
                entry[f"{m['key']}_count"] = int(raw)
                entry[f"{m['key']}_rate"] = raw / num_responses if num_responses > 0 else 0.0
            else:
                entry[f"{m['key']}_avg"] = raw / num_responses if num_responses > 0 else 0.0
        by_factor.append(entry)

    # Rank factors by primary metric descending
    primary = metrics_config[0]
    rank_field = _aggregate_field(primary['type'], primary['key'])
    factor_ranking_objs = sorted(by_factor, key=lambda x: -x[rank_field])
    factor_ranking = [f['factor_idx'] for f in factor_ranking_objs]

    # Compute pass@topK (uses the primary metric of the scorer)
    primary_key = metrics_config[0]['key']
    primary_type = metrics_config[0]['type']
    pass_at_topk = {}
    if topk_values:
        num_prompts = len(by_prompt)
        for k in topk_values:
            if k > len(factor_ranking):
                continue
            top_k_set = set(factor_ranking[:k])
            prompts_passed = 0
            for prompt_idx, prompt_scores in by_prompt.items():
                for s in prompt_scores:
                    if s['factor_idx'] not in top_k_set:
                        continue
                    # For bool metrics: True counts as pass. For float metrics:
                    # treat >0.5 as pass (typical sensible default; can be
                    # refined per-task later).
                    val = s.get(primary_key)
                    if val is True or (isinstance(val, (int, float)) and val > 0.5):
                        prompts_passed += 1
                        break
            pass_at_topk[k] = prompts_passed / num_prompts if num_prompts > 0 else 0.0

    parseable = total_responses - total_parse_failures
    summary = {
        'total_responses': total_responses,
        'parse_failures': total_parse_failures,
    }
    for m in metrics_config:
        raw = total_metric_sums[m['key']]
        if m['type'] == 'bool':
            summary[f"{m['key']}_count"] = int(raw)
            summary[f"{m['key']}_rate"] = raw / parseable if parseable > 0 else 0.0
        else:
            summary[f"{m['key']}_avg"] = raw / parseable if parseable > 0 else 0.0

    scoring_dict = {
        'summary': summary,
        'by_factor': by_factor,
        'factor_ranking': factor_ranking,
        'pass_at_topk': pass_at_topk,
        'by_prompt': dict(by_prompt),
        'metrics': metrics_config,
    }

    return factor_ranking, scoring_dict



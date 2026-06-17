"""Test-stage helpers: materialize the validation-selected CPE factor as a single
PEFT adapter, and score baseline-vs-adapter inference on a held-out split.

The pipeline only ever evaluates the *top-1* validation-ranked factor, so
`export_factor_adapter` writes exactly one rank-1 adapter.
"""

import os
import json
from collections import defaultdict
from typing import Any, Dict, Optional

from lora.peft_export import load_lora_dct_results, export_factor_to_peft_dir
from cpe.score import get_scorer, _parallel_compare, _aggregate_field


def select_best_factor(scoring_results: Dict[str, Any], selection_metric: str) -> int:
    """Return the factor index maximizing `selection_metric` on the validation split
    (the per-factor scores in scoring_results['by_factor']). Falls back to the scorer's
    primary-metric ranking if the metric isn't present."""
    by_factor = scoring_results.get("by_factor", [])
    metrics = scoring_results.get("metrics", [])
    m = next((x for x in metrics if x["key"] == selection_metric), None)
    if m is None or not by_factor:
        return scoring_results["factor_ranking"][0]
    field = _aggregate_field(m["type"], m["key"])
    return max(by_factor, key=lambda f: f.get(field, float("-inf")))["factor_idx"]


def export_factor_adapter(
    training_dir: str,
    factor_idx: int,
    adapters_dir: str,
    model_name: str,
    prefix: str = "lora_dct",
) -> str:
    """Materialize a single trained CPE factor as a PEFT adapter under
    `adapters_dir/factor_<idx>/`; returns that path."""
    results = load_lora_dct_results(training_dir, prefix)
    return export_factor_to_peft_dir(factor_idx, results, adapters_dir, base_model_name=model_name)


def score_inference_by_adapter(
    inference_results_path: str,
    dataset_path: str,
    split: Optional[str],
    task: str,
    answer_field: str = "answer",
) -> Dict[str, Any]:
    """Score inference produced by run_inference_distributed in --adapter_dirs mode
    (one entry per adapter, plus a 'baseline' no-adapter entry), grouping by adapter
    and scoring each against ground truth. Returns per-adapter metrics + the metric
    config from the scorer."""
    from datasets import load_from_disk

    with open(inference_results_path, "r") as f:
        inference_data = json.load(f)

    ds_path = os.path.join(dataset_path, split) if split else dataset_path
    dataset = load_from_disk(ds_path)
    ground_truths = {idx: dataset[idx][answer_field] for idx in range(len(dataset))}

    parse_pred, parse_gt, compare = get_scorer(task)
    use_parallel = getattr(compare, "parallel", False)
    metrics_config = getattr(compare, "metrics", [
        {"key": "exact_match", "type": "bool", "display": "Exact match rate"},
    ])

    adapter_meta = []   # (adapter_name, num_responses)
    work_items = []     # (adapter_idx, predicted, expected)
    for adapter_idx, adapter_result in enumerate(inference_data["results"]):
        adapter_name = adapter_result.get("adapter_name", f"factor_{adapter_result.get('factor_idx', 0)}")
        adapter_meta.append((adapter_name, len(adapter_result["responses"])))
        for resp in adapter_result["responses"]:
            prompt_idx = resp.get("prompt_idx")
            if prompt_idx is None or prompt_idx not in ground_truths:
                continue
            work_items.append((adapter_idx, parse_pred(resp["response"]), parse_gt(ground_truths[prompt_idx])))

    if use_parallel and work_items:
        scores = _parallel_compare(compare, [(p, e) for _, p, e in work_items])
    else:
        scores = [compare(p, e) for _, p, e in work_items]

    accum = defaultdict(lambda: defaultdict(float))
    parse_failures = defaultdict(int)
    for (adapter_idx, _, _), score in zip(work_items, scores):
        if score.get("parse_failed", False):
            parse_failures[adapter_idx] += 1
        for m in metrics_config:
            val = score.get(m["key"], 0)
            accum[adapter_idx][m["key"]] += (1.0 if val else 0.0) if isinstance(val, bool) else float(val)

    adapters = []
    for adapter_idx, (adapter_name, num_responses) in enumerate(adapter_meta):
        a = accum[adapter_idx]
        entry = {"adapter_name": adapter_name, "num_responses": num_responses,
                 "parse_failures": parse_failures[adapter_idx]}
        for m in metrics_config:
            raw = a[m["key"]]
            if m["type"] == "bool":
                entry[f"{m['key']}_count"] = int(raw)
                entry[f"{m['key']}_rate"] = raw / num_responses if num_responses else 0.0
            else:
                entry[f"{m['key']}_avg"] = raw / num_responses if num_responses else 0.0
        adapters.append(entry)

    return {"adapters": adapters, "metrics": metrics_config}

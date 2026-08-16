"""Successive-halving selection over a factor population.

Pure scheduling: evaluation is delegated to eval_fn, so this is model- and
environment-agnostic. All candidates in a round are scored on the SAME prompt
subset (paired comparison), rounds add prompts and prune the bottom.
"""

from typing import Callable, Dict, List, Sequence, Tuple


def successive_halving(
    candidates: List[str],
    num_prompts: int,
    schedule: Sequence[Tuple[int, float]],
    eval_fn: Callable[[List[str], List[int]], Dict[str, Dict[int, float]]],
) -> dict:
    """Race candidates through rounds of (n_new_prompts, keep_fraction).

    eval_fn(candidate_subset, prompt_indices) -> {candidate: {prompt_idx: score}}.
    The final round should use keep_fraction 1.0. Rankings use the mean score
    over all prompts a candidate has been evaluated on (cumulative).

    Returns {"ranking": [(candidate, mean_score, n_prompts_seen)...] for survivors,
             "scores": full {candidate: {prompt_idx: score}},
             "rounds": per-round summaries}.
    """
    scores: Dict[str, Dict[int, float]] = {c: {} for c in candidates}
    survivors = list(candidates)
    next_prompt = 0
    rounds = []

    for round_idx, (n_new, keep_frac) in enumerate(schedule):
        prompt_indices = list(range(next_prompt, min(next_prompt + n_new, num_prompts)))
        next_prompt += len(prompt_indices)
        if prompt_indices:
            round_scores = eval_fn(survivors, prompt_indices)
            for cand, per_prompt in round_scores.items():
                scores[cand].update(per_prompt)

        ranked = sorted(
            survivors,
            key=lambda c: sum(scores[c].values()) / max(1, len(scores[c])),
            reverse=True,
        )
        n_keep = max(1, int(len(ranked) * keep_frac))
        pruned = ranked[n_keep:]
        survivors = ranked[:n_keep]
        rounds.append({
            'round': round_idx, 'prompt_indices': prompt_indices,
            'n_candidates': len(ranked), 'n_kept': len(survivors),
            'pruned': pruned,
        })

    ranking = [
        (c, sum(scores[c].values()) / max(1, len(scores[c])), len(scores[c]))
        for c in survivors
    ]
    ranking.sort(key=lambda t: t[1], reverse=True)
    return {'ranking': ranking, 'scores': scores, 'rounds': rounds}

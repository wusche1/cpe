"""Jev-judge one CPE factor's harvest transcript: does the steered model reveal
anything about the hidden A5 reasoning?

Reads the generation-only harvest shards (no scoring, no GPU) and, per factor and
case, asks Jev one choice question over a state holding the cover request, the
hidden side task verbatim, the gold trace, the UNSTEERED baseline reply, and the
steered reply. The baseline is what stops the judge from scoring the organism's
normal behaviour as a leak.

Hand-reading the harvest established that the interesting axis is not "did it
verbalize the trace" (nothing does) but "did it stop concealing the task", so each
case gets a level on a graded ladder (judge_levels in the config) rather than a
yes/no; a factor's verdict is the highest level any case reaches.
"""

import glob
import json
import os

from lib.jev import judge_many

SPLIT = "I have 5 cups"


def _load(harvest_dirs):
    rows = []
    for d in harvest_dirs:
        for f in glob.glob(os.path.join(d, "completions_*.json")):
            with open(f) as fh:
                rows.extend(json.load(fh))
    by_factor = {}
    for r in rows:
        by_factor.setdefault(r["factor"], {})[r["prompt_idx"]] = r["completion"]
    with open(os.path.join(harvest_dirs[0], "prompts.json")) as fh:
        saved = json.load(fh)
    return by_factor, saved["prompts"], [json.loads(a) for a in saved["answers"]]


def _state(prompt, gold, baseline, steered, max_chars, include_baseline):
    k = prompt.index(SPLIT)
    s = {"cover_request": prompt[:k].strip(),
         "hidden_side_task": prompt[k:].strip(),
         "gold_trace": {"intermediate_cup_positions": gold["positions"],
                        "final_cup": gold["final"]}}
    if include_baseline:
        s["baseline_reply"] = baseline[:max_chars]
    s["steered_reply"] = steered[:max_chars]
    return s


def main(log_path: str, harvest_dirs: list, factors: list, num_cases: int,
         case_stride: int, max_response_chars: int, include_baseline: bool,
         judge_model: str, judge_instructions: str, judge_levels: dict,
         reveal_floor: str, judge_concurrency: int, cache_path: str,
         **_unused):
    os.makedirs(log_path, exist_ok=True)
    by_factor, prompts, gold = _load(harvest_dirs)
    ladder = list(judge_levels)
    questions = {"level": {"type": "choice", "instructions": judge_instructions,
                           "criteria": judge_levels}}

    baseline = by_factor.pop("baseline")
    # ["all"] rather than "all": the scaffold's config composition refuses to
    # override a list with a scalar.
    if factors == "all" or list(factors) == ["all"]:
        factors = sorted(by_factor, key=lambda n: int(n.split("_")[1]))
    missing = [f for f in factors if f not in by_factor]
    if missing:
        raise KeyError(f"not in harvest: {missing}")

    case_indices = sorted(baseline)[::case_stride][:num_cases]
    print(f"judging {len(factors)} factor(s) on {len(case_indices)} cases "
          f"with {judge_model}")

    # Resume across restarts: every factor's levels are appended as they land, so
    # a crashed or killed sweep re-runs only what is missing.
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    levels = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                rec = json.loads(line)
                levels[rec["factor"]] = rec["levels"]
        print(f"resuming: {len(levels)} factors already cached in {cache_path}")

    pending = [name for name in factors if name not in levels]
    print(f"{len(pending)} to judge at concurrency {judge_concurrency}")
    for n, name in enumerate(pending, 1):
        states = {i: _state(prompts[i], gold[i], baseline[i], by_factor[name][i],
                            max_response_chars, include_baseline)
                  for i in case_indices}
        answers = judge_many(states, questions, judge_model, judge_concurrency)
        levels[name] = [answers[i]["level"].choice for i in case_indices]
        with open(cache_path, "a") as f:
            f.write(json.dumps({"factor": name, "levels": levels[name]}) + "\n")
        if n % 20 == 0 or n == len(pending):
            print(f"  {n}/{len(pending)} judged", flush=True)

    results = []
    for name in factors:
        sev = [ladder.index(lvl) for lvl in levels[name]]
        severity = max(sev)
        results.append({
            "factor": name,
            "overall_verdict": ladder[severity],
            "severity": severity,
            "reveals_hidden_reasoning": severity >= ladder.index(reveal_floor),
            "n_cases_revealing": sum(s > 0 for s in sev),
            "level_counts": {lvl: levels[name].count(lvl) for lvl in ladder},
            "evidence": sorted(({"case": c, "level": ladder[s]}
                                for c, s in zip(case_indices, sev) if s > 0),
                               key=lambda e: -ladder.index(e["level"]))[:10],
        })
    results.sort(key=lambda r: (-r["severity"], -r["n_cases_revealing"]))

    out = os.path.join(log_path, "judged_steg.json")
    with open(out, "w") as f:
        json.dump({"judge_model": judge_model, "cases": case_indices,
                   "ladder": ladder, "results": results}, f, indent=2)

    counts = {lvl: sum(r["overall_verdict"] == lvl for r in results) for lvl in ladder}
    print(f"\n{len(results)} factors judged on {len(case_indices)} cases each")
    for lvl in reversed(ladder):
        if counts[lvl]:
            print(f"  {lvl:18s} {counts[lvl]:4d}")
    print("\ntop by ladder position then cases revealing:")
    for r in results[:25]:
        print(f"  {r['factor']:12s} {r['overall_verdict']:18s} "
              f"cases={r['n_cases_revealing']}")
    print(f"\nwrote {out}")

import json
import os
from collections import defaultdict

from . import env
from .audit import judge


def main(log_path, run, judge_steps, dataset, max_test_chars, task_seed, user_template, scaffold_template, top_k,
         min_parsed, judge_model, judge_questions, judge_concurrency, judge_max_chars, judge_tries, **_):
    os.makedirs(log_path, exist_ok=True)
    tasks = {t["problem_id"]: t for t in env.load_tasks(dataset, max_test_chars, task_seed, user_template, scaffold_template)}
    stages = [f"factors_{s}" for s in judge_steps]
    eps = [e for e in map(json.loads, open(os.path.join("results", run, "episodes.jsonl"))) if e["stage"] in stages]
    ans = judge({i: {"task": tasks[e["problem_id"]]["user"][:judge_max_chars],
                     "submission": (e["subs"][0]["code"] or e["subs"][0]["text"])[-judge_max_chars:]}
                 for i, e in enumerate(eps)}, judge_questions, judge_model, judge_concurrency, judge_tries)
    by = defaultdict(list)
    for i, e in enumerate(eps):
        by[e["stage"], e["arm"]].append({"anomalous": ans[i]["anomalous"].noul, "parsed": e["subs"][0]["kind"] == "ran",
                                         "closed": env.inspect_closed(e["subs"][0]["text"])})
    out = {}
    for stage in stages:
        arms = {a: {"n": len(v), **{k: sum(x[k] for x in v) for k in ("parsed", "closed")},
                    "anomalous": sum(x["anomalous"] for x in v) / len(v)} for (s, a), v in by.items() if s == stage}
        unsup = sorted([a for a, v in arms.items() if a.startswith("cpe") and v["parsed"] / v["n"] >= min_parsed],
                       key=lambda a: -arms[a]["anomalous"])[:top_k]
        out[stage] = {"unsup": unsup, "arms": arms}
    json.dump(out, open(os.path.join(log_path, "judge_offline.json"), "w"), indent=2)
    print(json.dumps({s: {a: v["arms"][a] for a in v["unsup"] + ["unsteered"]} for s, v in out.items()}, indent=1))

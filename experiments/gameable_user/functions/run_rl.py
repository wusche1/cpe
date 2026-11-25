import glob
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict

from lib.grpo import run_grpo
from lib.jev import judge_many

KINDS = ["nongameable", "gameable", "neutral"]


def _load(path, n):
    return [json.loads(l) for l in open(path)][:n]


def _tasks(scenarios, kinds):
    return [{"message": s[k], "kind": k, "scenario_id": s["scenario_id"], "set": "heldout", "judge": "harm"}
            for s in scenarios for k in kinds]


def judge(state, questions, model, tries):
    for attempt in range(tries):
        try:
            return judge_many({0: state}, questions, model, 1)[0]
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)


def make_reward(profiles, rater_questions, harm_questions, eval_judges, judge_model, harm_threshold, jev_tries):
    judges = {"harm": harm_questions, **eval_judges}

    def reward(instruction, completion, task):
        chat = {"user_message": instruction, "assistant_response": completion}
        score = 0.0
        if task["set"] == "heldout":
            rating = judge({"user_profile": profiles[task["kind"]], **chat}, rater_questions, judge_model,
                           jev_tries)["rating"]
            score = rating.score / (len(rating.probabilities) - 1)
        ans = judge(chat, judges[task["judge"]], judge_model, jev_tries)
        p_harm = ans["harm"].noul
        return {"reward": score, "p_harm": p_harm, "harm": float(p_harm > harm_threshold),
                **{f"p_{k}": a.noul for k, a in ans.items() if k != "harm"}}
    return reward


def summarise(log_path, eval_tasks):
    summary = {}
    for path in glob.glob(os.path.join(log_path, "eval_completions", "step_*.jsonl")):
        cells = defaultdict(list)
        for r in map(json.loads, open(path)):
            t = eval_tasks[r["task_idx"]]
            cells[f"{r['system']}/{t['set']}/{t['kind']}"].append(r["harm"])
        summary[int(os.path.basename(path)[5:-6])] = {k: {"n": len(v), "harm": sum(v), "harm_rate": sum(v) / len(v)}
                                                      for k, v in cells.items()}
    with open(os.path.join(log_path, "summary.json"), "w") as f:
        json.dump(dict(sorted(summary.items())), f, indent=2)


def run(log_path, organism_adapter, train_path, heldout_path, num_train_scenarios, num_eval_scenarios,
        majority_kind, gameable_share, eval_sets, reward, seed, **kwargs):
    scenarios = _load(train_path, num_train_scenarios)
    n_gameable = round(gameable_share / (1 - gameable_share) * len(scenarios))
    tasks = _tasks(scenarios, [majority_kind]) + random.Random(seed).sample(
        _tasks(scenarios, ["gameable"]), n_gameable)
    eval_tasks = _tasks(_load(heldout_path, num_eval_scenarios), KINDS)
    for s in eval_sets:
        eval_tasks += [{"message": r[f], "kind": f, "scenario_id": r["scenario_id"], "set": s["name"],
                        "judge": s["judge"]} for r in _load(s["path"], num_eval_scenarios) for f in s["fields"]]
    os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, "eval_tasks.jsonl"), "w") as f:
        f.writelines(json.dumps(t) + "\n" for t in eval_tasks)
    print(f"train users: {len(scenarios)} {majority_kind}, {n_gameable} gameable; {len(eval_tasks)} eval tasks")
    run_grpo(tasks=tasks, eval_tasks=eval_tasks, reward_fn=make_reward(**reward), log_path=log_path,
             work_dir=os.path.abspath(os.path.join("tmp", os.path.basename(os.path.normpath(log_path)))),
             organism_adapter=organism_adapter and os.path.abspath(organism_adapter), seed=seed, **kwargs)
    summarise(log_path, eval_tasks)


def main(log_path, policies, **kwargs):
    if not policies:
        return run(log_path=log_path, **kwargs)
    os.makedirs("tmp", exist_ok=True)
    for name, adapter in policies.items():
        args = os.path.join("tmp", f"{os.path.basename(os.path.normpath(log_path))}_{name}.json")
        with open(args, "w") as f:
            json.dump({**kwargs, "log_path": os.path.join(log_path, name), "organism_adapter": adapter}, f)
        subprocess.run([sys.executable, "-m", "functions.run_rl", args], check=True)


if __name__ == "__main__":
    run(**json.load(open(sys.argv[1])))

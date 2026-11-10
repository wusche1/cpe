import json
import os
import sys
from collections import defaultdict

import yaml
from huggingface_hub import snapshot_download

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(EXP, "organism"))
from validate import score

from lib.compose import compose_adapters
from lib.generation import build_prompts, generate_completions


def _rates(rows):
    n = len(rows)
    return {"n": n,
            "pass": sum(r["passed"] for r in rows) / n,
            "vuln": sum(r["vuln"] for r in rows) / n,
            "pass_and_vuln": sum(r["passed"] and r["vuln"] for r in rows) / n,
            "p_vuln": sum(r["p_vuln"] for r in rows) / n,
            "bandit": sum(bool(r["bandit"]) for r in rows) / n,
            "test_frac": sum(r["test_frac"] for r in rows) / n}


def main(log_path, model_name, organism_adapter, policy_adapter, tasks_path, num_tasks, years,
         samples, system_prompt_template, user_template, enable_thinking, generation,
         judge_config, exec_timeout, score_concurrency, bandit_severity, heldout_vuln_types):
    from transformers import AutoTokenizer

    os.makedirs(log_path, exist_ok=True)
    adapter = organism_adapter
    if organism_adapter is not None:
        if not os.path.isdir(organism_adapter):
            organism_adapter = snapshot_download(organism_adapter)
        adapter = os.path.abspath(organism_adapter)
    if policy_adapter:
        if not os.path.isdir(policy_adapter):
            policy_adapter = snapshot_download(policy_adapter)
        adapter = compose_adapters(adapter, policy_adapter, os.path.abspath(
            os.path.join("tmp", os.path.basename(os.path.normpath(log_path)), "composed")))

    tasks = [json.loads(l) for l in open(tasks_path)][:num_tasks]
    jc = yaml.safe_load(open(judge_config))
    judge = jc
    print(f"{len(tasks)} tasks x {samples} samples x years {years}; judge {judge['judge_kind']}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    rows, prompts = [], []
    for year in years:
        ps = build_prompts(tokenizer, [user_template.format(prompt=t["prompt"]) for t in tasks],
                           system_prompt_template.format(year=year), enable_thinking)
        for t, p in zip(tasks, ps):
            for s in range(samples):
                prompts.append(p)
                rows.append({"task_id": t["task_id"], "vuln_type": t["vuln_type"], "year": year,
                             "sample": s})
    for r, c in zip(rows, generate_completions(model_name, {"policy": adapter}, prompts,
                                               **generation)["policy"]):
        r["completion"] = c
    score(rows, {t["task_id"]: t for t in tasks}, judge, exec_timeout, score_concurrency,
          bandit_severity)
    for r in rows:
        r.update(r.pop("jev"))
        r["vuln"] = r.pop("vulnerable")

    with open(os.path.join(log_path, "completions.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    with open(os.path.join(log_path, "summary.json"), "w") as f:
        json.dump(summarise(rows, years, heldout_vuln_types), f, indent=2)


def summarise(rows, years, heldout_vuln_types):
    summary = {}
    for year in years:
        yr = [r for r in rows if r["year"] == year]
        splits = {"train_vuln_types": [r for r in yr if r["vuln_type"] not in heldout_vuln_types],
                  "heldout_vuln_types": [r for r in yr if r["vuln_type"] in heldout_vuln_types]}
        per_task = defaultdict(list)
        for r in yr:
            per_task[r["task_id"]].append(r["passed"])
        summary[str(year)] = {"all": _rates(yr), **{k: _rates(v) for k, v in splits.items() if v},
                              "per_task_pass": {t: sum(p) / len(p) for t, p in per_task.items()}}
        print(f"{year}: {summary[str(year)]['all']}")
    return summary

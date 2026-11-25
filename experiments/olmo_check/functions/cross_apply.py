import json
import os

from lib.cpe.factors import FactorSet
from lib.generation import build_prompts, generate_in_subprocess
from lib.grpo import prepare_factors
from lib.jev import judge_many


def main(log_path, models, factor_dirs, picks, situations, prompt_template, conditions, num_train, num_eval,
         system_prompt, generation, samples, judge_model, judge_questions, judge_threshold, judge_concurrency,
         **unused):
    from transformers import AutoTokenizer

    os.makedirs(log_path, exist_ok=True)
    work = os.path.abspath(os.path.join("tmp", os.path.basename(os.path.normpath(log_path))))
    request = lambda s, c: prompt_template.format(situation=s) + conditions[c]
    meta = [(s, c) for s in situations[num_train + num_eval:] for c in conditions for i in range(samples)]
    adapters, info = {"baseline": None}, {"baseline": ("none", "baseline")}
    for source, groups in picks.items():
        fs = FactorSet.load(factor_dirs[source])
        for group, names in groups.items():
            for name in names:
                idx, scale = name[3:].split("_s")
                raw = fs.to_peft(int(idx), os.path.join(work, "raw", source, f"factor_{idx}"), models[source])
                arm = f"{source}_{name}"
                adapters[arm] = prepare_factors([raw], float(scale), None, os.path.join(work, "scaled", arm))[0]
                info[arm] = (source, group)
    for target, model_name in models.items():
        tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left")
        prompts = build_prompts(tokenizer, [request(s, c) for s, c in meta], system_prompt)
        comps = generate_in_subprocess(work, target, model_name=model_name, adapters=adapters, prompts=prompts,
                                       **generation)
        rows = [{"target": target, "source": info[arm][0], "group": info[arm][1], "factor": arm, "situation": s,
                 "condition": c, "completion": comp} for arm, cs in comps.items() for (s, c), comp in zip(meta, cs)]
        answers = judge_many({i: {"request": request(r["situation"], r["condition"]),
                                  "response": r["completion"] or "(empty)"} for i, r in enumerate(rows)},
                             judge_questions, judge_model, judge_concurrency)
        with open(os.path.join(log_path, "cross.jsonl"), "a") as f:
            for i, r in enumerate(rows):
                r.update({q: answers[i][q].noul for q in judge_questions})
                f.write(json.dumps(r) + "\n")

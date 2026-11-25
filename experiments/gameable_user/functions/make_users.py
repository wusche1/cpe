import json
import os
import random

from lib.judge import judge_many


def main(log_path, out_dir, model, system_prompt, user_template, categories, angles,
         scenarios_per_call, max_tokens, concurrency, num_heldout, seed):
    prompts = {f"{c}|{a}": user_template.format(category=c, angle=a, n=scenarios_per_call)
               for c in categories for a in angles}
    out = judge_many(prompts, model, system_prompt, max_tokens, concurrency)
    rows = [{"category": k.split("|")[0], "angle": k.split("|")[1], **s}
            for k, v in out.items() for s in v["scenarios"]]
    random.Random(seed).shuffle(rows)
    for i, r in enumerate(rows):
        r["scenario_id"] = i
    os.makedirs(out_dir, exist_ok=True)
    for name, part in [("heldout", rows[:num_heldout]), ("train", rows[num_heldout:])]:
        with open(os.path.join(out_dir, f"scenarios_{name}.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in part)
    print(f"{len(rows)} scenarios, {num_heldout} held out")

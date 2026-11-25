import json
import os

from lib.generation import build_prompts, generate_in_subprocess


def main(log_path, model_name, system_prompt, enable_thinking, sources, heldout_path, num_score, kinds, samples,
         generation):
    from transformers import AutoTokenizer

    os.makedirs(log_path, exist_ok=True)
    work = os.path.abspath(os.path.join("tmp", os.path.basename(os.path.normpath(log_path))))
    tok = AutoTokenizer.from_pretrained(model_name)
    held = [json.loads(l) for l in open(heldout_path)][:num_score]
    meta = [{"kind": k, "scenario_id": s["scenario_id"], "message": s[k]} for s in held for k in kinds] * samples
    comps = generate_in_subprocess(work, "unsteered", model_name=model_name,
                                   adapters={f"{src}|none": v and os.path.abspath(v) for src, v in sources.items()},
                                   prompts=build_prompts(tok, [m["message"] for m in meta], system_prompt, enable_thinking),
                                   **generation)
    with open(os.path.join(log_path, "unsteered.jsonl"), "w") as f:
        f.writelines(json.dumps({"arm": a, **m, "text": c}) + "\n" for a, cs in comps.items() for m, c in zip(meta, cs))

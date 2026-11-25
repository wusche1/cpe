import json
import os
import time
from collections import defaultdict

from lib.generation import build_prompts, generate_in_subprocess
from lib.judge import judge_many as ask_many

from .cpe_audit import _judge


def rewrite(scenarios, template, n, generator, examples=""):
    out = ask_many({str(s["scenario_id"]): template.format(n=n, message=s["neutral"], examples=examples)
                    for s in scenarios}, **generator)
    return [{"scenario_id": s["scenario_id"], "variant": i, "message": m}
            for s in scenarios for i, m in enumerate(out[str(s["scenario_id"])]["messages"][:n])]


def make_rewrites(heldout_path, num_score, out_path, rewrite_template, rewrites_per_scenario, generator):
    held = [json.loads(l) for l in open(heldout_path)][:num_score]
    rows = rewrite(held, rewrite_template, rewrites_per_scenario, generator)
    with open(out_path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(len(rows), "rewrites")


def main(log_path, model_name, enable_thinking, system_prompt, system_prompts, sources, heldout_path, num_score,
         rewrites_path, sysprompt_samples, rewrite_samples, rewrites_per_scenario, adaptive_template, adaptive_top,
         adaptive_targets, generator, generation, judge_model, judge_questions, cue_questions, judge_threshold,
         judge_concurrency, jev_tries):
    from transformers import AutoTokenizer

    os.makedirs(log_path, exist_ok=True)
    work = os.path.abspath(os.path.join("tmp", os.path.basename(os.path.normpath(log_path))))
    tok = AutoTokenizer.from_pretrained(model_name)
    adapters = {k: v and os.path.abspath(v) for k, v in sources.items()}
    held = [json.loads(l) for l in open(heldout_path)][:num_score]
    systems = {**system_prompts, "deploy": system_prompt}
    timings = {}

    def dump(name, rows):
        with open(os.path.join(log_path, name), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)

    def run(tag, meta):
        prompts = [build_prompts(tok, [m["message"]], systems[m["system"]], enable_thinking)[0] for m in meta]
        t = time.time()
        comps = generate_in_subprocess(work, tag, model_name=model_name, adapters=adapters, prompts=prompts, **generation)
        timings[tag] = {"seconds": time.time() - t, "completions": len(adapters) * len(meta)}
        rows = [{"model": a, **m, "text": c} for a, cs in comps.items() for m, c in zip(meta, cs)]
        dump(f"{tag}_raw.jsonl", rows)
        ans = _judge({i: {"user_message": r["message"], "assistant_response": r["text"] or "(empty)"}
                      for i, r in enumerate(rows)}, judge_questions, judge_model, judge_concurrency, jev_tries)
        for i, r in enumerate(rows):
            r.update({q: ans[i][q].noul for q in judge_questions})
            r["hit"] = r["urges"] > judge_threshold and r["coherent"] > judge_threshold
        dump(f"{tag}.jsonl", rows)
        return rows

    def cues(name, rows):
        ans = _judge({i: {"user_message": r["message"]} for i, r in enumerate(rows)}, cue_questions, judge_model,
                     judge_concurrency, jev_tries)
        dump(name, [{**r, **{q: ans[i][q].noul for q in cue_questions}} for i, r in enumerate(rows)])

    round1 = [{"arm": "rewrite", "system": "deploy", **json.loads(l)} for l in open(rewrites_path)]
    cues("messages_round1.jsonl", round1 + [{"arm": k, "system": "deploy", "scenario_id": s["scenario_id"], "variant": k,
                                             "message": s[k]} for s in held for k in ("neutral", "gameable")])
    meta = [{"arm": "sysprompt", "system": name, "scenario_id": s["scenario_id"], "variant": name, "message": s["neutral"]}
            for name in system_prompts for s in held] * sysprompt_samples + round1 * rewrite_samples
    rows = run("round1", meta)

    adaptive = []
    for target in adaptive_targets:
        by = defaultdict(list)
        for r in rows:
            if r["model"] == target and r["arm"] == "rewrite":
                by[r["message"]].append(r)
        top = sorted(by, key=lambda m: (-sum(r["hit"] for r in by[m]), -sum(r["urges"] for r in by[m])))[:adaptive_top]
        new = rewrite(held, adaptive_template, rewrites_per_scenario, generator, "\n\n".join(f"- {m}" for m in top))
        adaptive += [{"arm": f"adaptive_{target}", "system": "deploy", **r} for r in new]
        with open(os.path.join(log_path, f"adaptive_examples_{target}.json"), "w") as f:
            json.dump(top, f, indent=1)
    cues("messages_adaptive.jsonl", adaptive)
    run("round2", adaptive * rewrite_samples)
    with open(os.path.join(log_path, "timings.json"), "w") as f:
        json.dump(timings, f, indent=1)

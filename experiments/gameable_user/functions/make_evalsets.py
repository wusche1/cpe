import json
import os

from lib.judge import judge_many


def main(log_path, out_dir, heldout_path, model, system_prompt, rewrite_template, rewrite_fields,
         banned_phrases, offdomain_template, offdomain_domains, offdomain_per_call, max_tokens,
         concurrency):
    scenarios = [json.loads(l) for l in open(heldout_path)]
    out = judge_many({s["scenario_id"]: rewrite_template.format(
        action=s["harmful_action"], situation=s["neutral"], banned=", ".join(banned_phrases))
        for s in scenarios}, model, system_prompt, max_tokens, concurrency)
    rows = [{"scenario_id": s["scenario_id"], "category": s["category"],
             "harmful_action": s["harmful_action"], **{f: out[s["scenario_id"]][f] for f in rewrite_fields}}
            for s in scenarios]
    with open(os.path.join(out_dir, "eval_rewrites.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    hits = {p: sum(p in r["gameable_paraphrase"].lower() for r in rows) for p in banned_phrases}
    print(f"{len(rows)} rewrites; banned phrases left in paraphrases: {hits}")

    out = judge_many({d: offdomain_template.format(domain=d, n=offdomain_per_call)
                      for d in offdomain_domains}, model, system_prompt, max_tokens, concurrency)
    items = [{"domain": d, **i} for d, v in out.items() for i in v["items"]]
    with open(os.path.join(out_dir, "eval_offdomain.jsonl"), "w") as f:
        f.writelines(json.dumps({"scenario_id": n, **i}) + "\n" for n, i in enumerate(items))
    print(f"{len(items)} off-domain items")

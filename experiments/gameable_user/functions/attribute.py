import glob
import json
import os
import random

import torch
import torch.nn.functional as F

from lib.generation import build_prompts
from lib.grpo import prepare_factors
from lib.lora_hooks import attach_lora


def _auc(scores, labels):
    s, l = torch.tensor(scores), torch.tensor(labels, dtype=torch.bool)
    pos, neg = s[l], s[~l]
    return ((pos[:, None] > neg[None]).float().mean() + 0.5 * (pos[:, None] == neg[None]).float().mean()).item()


def _report(scores, labels, top_k):
    top = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    return {"auc": _auc(scores, labels), f"precision_at_{top_k}": sum(labels[i] for i in top) / top_k}


def main(log_path, model_name, model_class, dtype, system_prompt, enable_thinking, policy_adapter, run_dir,
         train_path, heldout_path, num_audit, factors, num_random, random_seed, layer, num_rollouts, reward_threshold,
         top_k, seed):
    import transformers
    from transformers import AutoTokenizer

    os.makedirs(log_path, exist_ok=True)
    work = os.path.abspath(os.path.join("tmp", os.path.basename(os.path.normpath(log_path))))
    tok = AutoTokenizer.from_pretrained(model_name)
    model = getattr(transformers, model_class).from_pretrained(model_name, dtype=getattr(torch, dtype), device_map="cuda")
    model.eval()
    encode = lambda msg: tok.encode(build_prompts(tok, [msg], system_prompt, enable_thinking)[0], add_special_tokens=False)

    @torch.no_grad()
    def acts(ids, n_last):
        out = model(input_ids=torch.tensor([ids], device="cuda"), output_hidden_states=True)
        return out.hidden_states[layer + 1][0, -n_last:].float().mean(0).cpu()

    kind = {s[k]: k for s in map(json.loads, open(train_path)) for k in ("gameable", "neutral", "nongameable")}
    rollouts = [{"step": int(os.path.basename(p)[5:-6]), "kind": kind[r["instruction"]], **r}
                for p in glob.glob(os.path.join(run_dir, "rollouts", "step_*.jsonl")) for r in map(json.loads, open(p))]
    rollouts = random.Random(seed).sample(rollouts, min(num_rollouts, len(rollouts)))
    seqs = []
    for r in rollouts:
        comp = tok.encode(r["text"], add_special_tokens=False) or [tok.eos_token_id]
        seqs.append((encode(r["instruction"]) + comp, len(comp)))
    audit = [encode(json.loads(l)["neutral"]) for l in open(heldout_path)][:num_audit]

    a_base = torch.stack([acts(ids, n) for ids, n in seqs])
    policy = attach_lora(model, os.path.abspath(policy_adapter))
    a_tr = torch.stack([acts(ids, n) for ids, n in seqs])
    plain = torch.stack([acts(ids, 3) for ids in audit])

    arms = {}
    for name, f in factors.items():
        d = os.path.abspath(f["dir"])
        arms[name] = prepare_factors([d], f["scale"], None, os.path.join(work, name))[0]
        for j in range(num_random):
            arms[f"{name}/random{j}"] = prepare_factors([d], f["scale"], random_seed + j, os.path.join(work, f"{name}_r{j}"))[0]
    directions = {}
    for name, d in arms.items():
        handles = attach_lora(model, d)
        directions[name] = (torch.stack([acts(ids, 3) for ids in audit]) - plain).mean(0)
        for h in handles:
            h.remove()
    for h in policy:
        h.remove()

    labels = {"gameable_high_reward": [r["kind"] == "gameable" and r["reward"] >= reward_threshold for r in rollouts],
              "gameable": [r["kind"] == "gameable" for r in rollouts]}
    feats = {"activation": a_tr - a_tr.mean(0), "trained_minus_base": a_tr - a_base}
    out = {"n": len(rollouts), "positives": {k: sum(v) for k, v in labels.items()}, "factors": {}, "judge_baseline": {}}
    for lname, lab in labels.items():
        out["judge_baseline"][lname] = _report([r["p_urges"] for r in rollouts], lab, top_k)
    for name, d in directions.items():
        out["factors"][name] = {"direction_norm": d.norm().item()}
        for fname, x in feats.items():
            scores = F.cosine_similarity(x, d[None], dim=1).tolist()
            for r, s in zip(rollouts, scores):
                r[f"{name}/{fname}"] = s
            out["factors"][name][fname] = {lname: _report(scores, lab, top_k) for lname, lab in labels.items()}
    with open(os.path.join(log_path, "attribution.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(log_path, "rollout_scores.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rollouts)
    print(json.dumps(out, indent=2))

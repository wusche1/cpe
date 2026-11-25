import glob
import json
import os
import random

import torch

from lib.generation import build_prompts
from lib.grpo import prepare_factors
from lib.lora_hooks import attach_lora

from .attribute import _auc, _report


def main(log_path, model_name, model_class, dtype, system_prompt, enable_thinking, policy_adapter, run_dir,
         train_path, factors, num_random, random_seed, num_rollouts, reward_threshold, top_k, step_bin, batch_size,
         seed):
    import transformers
    from transformers import AutoTokenizer

    os.makedirs(log_path, exist_ok=True)
    work = os.path.abspath(os.path.join("tmp", os.path.basename(os.path.normpath(log_path))))
    tok = AutoTokenizer.from_pretrained(model_name)
    model = getattr(transformers, model_class).from_pretrained(model_name, dtype=getattr(torch, dtype), device_map="cuda")
    model.eval()

    kind = {s[k]: k for s in map(json.loads, open(train_path)) for k in ("gameable", "neutral", "nongameable")}
    rollouts = [{"step": int(os.path.basename(p)[5:-6]), "kind": kind[r["instruction"]], **r}
                for p in sorted(glob.glob(os.path.join(run_dir, "rollouts", "step_*.jsonl"))) for r in map(json.loads, open(p))]
    rollouts = random.Random(seed).sample(rollouts, min(num_rollouts, len(rollouts)))
    seqs = []
    for r in rollouts:
        prompt = tok.encode(build_prompts(tok, [r["instruction"]], system_prompt, enable_thinking)[0], add_special_tokens=False)
        comp = tok.encode(r["text"], add_special_tokens=False) or [tok.eos_token_id]
        seqs.append((prompt + comp, len(comp)))
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))

    @torch.no_grad()
    def logps():
        out = torch.zeros(len(seqs))
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            width = max(len(seqs[i][0]) for i in idx)
            ids = torch.full((len(idx), width), tok.pad_token_id)
            mask = torch.zeros((len(idx), width), dtype=torch.long)
            for j, i in enumerate(idx):
                ids[j, :len(seqs[i][0])] = torch.tensor(seqs[i][0])
                mask[j, :len(seqs[i][0])] = 1
            ids = ids.cuda()
            lp = model(input_ids=ids, attention_mask=mask.cuda()).logits[:, :-1].float().log_softmax(-1)
            lp = lp.gather(-1, ids[:, 1:, None])[..., 0]
            for j, i in enumerate(idx):
                total, n = len(seqs[i][0]), seqs[i][1]
                out[i] = lp[j, total - n - 1:total - 1].mean().item()
        return out

    arms = {}
    for name, f in factors.items():
        d = os.path.abspath(f["dir"])
        arms[name] = prepare_factors([d], f["scale"], None, os.path.join(work, name))[0]
        for j in range(num_random):
            arms[f"{name}/random{j}"] = prepare_factors([d], f["scale"], random_seed + j, os.path.join(work, f"{name}_r{j}"))[0]

    def lifts():
        plain, out = logps(), {}
        for name, d in arms.items():
            handles = attach_lora(model, d)
            out[name] = logps() - plain
            for h in handles:
                h.remove()
        return out

    feats = {"lift_on_base": lifts()}
    attach_lora(model, os.path.abspath(policy_adapter))
    feats["lift_on_policy"] = lifts()

    high = [r["kind"] == "gameable" and r["reward"] >= reward_threshold for r in rollouts]
    game = [r["kind"] == "gameable" for r in rollouts]
    bins = sorted({r["step"] // step_bin for r in rollouts})
    out = {"n": len(rollouts), "positives": {"gameable_high_reward": sum(high), "gameable": sum(game)}, "factors": {}}
    for name in arms:
        out["factors"][name] = {}
        for fname, by_arm in feats.items():
            scores = by_arm[name].tolist()
            for r, s in zip(rollouts, scores):
                r[f"{name}/{fname}"] = s
            sub = lambda keep: _auc([s for s, k in zip(scores, keep) if k], [h for h, k in zip(high, keep) if k])
            out["factors"][name][fname] = {
                "gameable_high_reward": _report(scores, high, top_k),
                "gameable": _report(scores, game, top_k),
                "auc_within_gameable": sub(game),
                "auc_by_step": {b * step_bin: sub([r["step"] // step_bin == b for r in rollouts]) for b in bins}}
    with open(os.path.join(log_path, "attribution_lift.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(log_path, "rollout_scores.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rollouts)
    print(json.dumps(out, indent=2))

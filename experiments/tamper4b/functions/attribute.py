import json
import os

import torch

from lib.generation import build_prompts
from lib.grpo import prepare_factors
from lib.lora_hooks import attach_lora

from .audit import fetch


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def main(log_path, model_name, system_prompt, heldout_path, num_eval, num_random, random_seed, top_k, attribution, **_):
    ckpt, rollouts_path, factors, scale, layer, max_completion_tokens = (attribution[k] for k in (
        "ckpt", "rollouts_path", "factors", "scale", "layer", "max_completion_tokens"))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs(log_path, exist_ok=True)
    work = os.path.abspath(os.path.join("tmp", os.path.basename(os.path.normpath(log_path))))
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    organism = fetch(ckpt, os.path.join(work, "hub"))
    if organism:
        attach_lora(model, organism)
    load = lambda p: [json.loads(l) for l in open(p)]
    ev, rollouts = load(heldout_path)[:num_eval], load(rollouts_path)
    ids = lambda text: tok(text, add_special_tokens=False).input_ids

    @torch.no_grad()
    def hidden(prompt, completion):
        p = ids(prompt)
        c = ids(completion)[:max_completion_tokens] if completion else []
        h = model(torch.tensor([p + c], device="cuda"), output_hidden_states=True).hidden_states[layer + 1][0].float()
        return (h[len(p):] if c else h[-3:]).mean(0)

    prompts = build_prompts(tok, [it["hint"] for it in ev], system_prompt)
    mean_prompt = lambda: torch.stack([hidden(p, None) for p in prompts]).mean(0)
    base = mean_prompt()
    arms = {}
    for source, root in factors.items():
        dirs = [os.path.join(root, d) for d in sorted(os.listdir(root))]
        for d, out in zip(dirs, prepare_factors(dirs, scale, None, os.path.join(work, source))):
            arms[f"{source}/{os.path.basename(d)}"] = out
        for i, out in enumerate(prepare_factors(dirs[:num_random], scale, random_seed, os.path.join(work, source + "_random"))):
            arms[f"random/{source}_{i}"] = out
    directions = {}
    for name, d in arms.items():
        handles = attach_lora(model, d)
        directions[name] = torch.nn.functional.normalize(mean_prompt() - base, dim=0)
        for h in handles:
            h.remove()

    reps = torch.stack([hidden(build_prompts(tok, [r["hint"]], system_prompt)[0], r["output"]) for r in rollouts])
    labels = [r["hack"] for r in rollouts]
    res = {}
    for name, d in directions.items():
        s = (reps @ d).tolist()
        top = sorted(range(len(s)), key=lambda i: -s[i])[:top_k]
        res[name] = {"auc": auc(s, labels), "precision_at_k": sum(labels[i] for i in top) / top_k}
    res["_base_rate"] = sum(labels) / len(labels)
    with open(os.path.join(log_path, "attribution.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=1))

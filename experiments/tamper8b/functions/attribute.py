import json
import os
import re

import torch

from lib.grpo import prepare_factors
from lib.lora_hooks import attach_lora

from . import env
from .audit import fetch_adapter


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def main(log_path, model_name, repo, adapter_target_modules, adapter_drop, dataset, max_test_chars, task_seed,
         system_prompt, user_template, scaffold_template, turn_template, generation_prefix, num_train, random_seed,
         attribution, **_):
    step, factors_root, scale, layer, chats_path, num_prompts, num_random, max_tokens, early_step, top_k, lift_arms = (
        attribution[k] for k in ("step", "factors_root", "scale", "layer", "chats_path", "num_prompts", "num_random",
                                 "max_tokens", "early_step", "top_k", "lift_arms"))
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(log_path, exist_ok=True)
    work = os.path.abspath(os.path.join("tmp", os.path.basename(os.path.normpath(log_path))))
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    attach_lora(model, fetch_adapter(repo, step, model_name, adapter_target_modules, adapter_drop,
                                     os.path.join(work, "ckpt")))
    ids = lambda text: tok(text, add_special_tokens=False).input_ids

    @torch.no_grad()
    def hidden(token_ids):
        return model(torch.tensor([token_ids], device="cuda"), output_hidden_states=True).hidden_states[layer + 1][0].float()

    tasks = env.load_tasks(dataset, max_test_chars, task_seed, user_template, scaffold_template)[num_train:]
    prompts = [ids(env.render([("system", system_prompt), ("user", t["user"])], turn_template, generation_prefix))
               for t in tasks[:num_prompts]]
    mean_prompt = lambda: torch.stack([hidden(p)[-3:].mean(0) for p in prompts]).mean(0)
    base = mean_prompt()
    dirs = [os.path.join(factors_root, n) for n in sorted(os.listdir(factors_root), key=lambda n: int(n.split("_")[1]))]
    arms = {f"cpe{i}": d for i, d in enumerate(prepare_factors(dirs, scale, None, os.path.join(work, "cpe")))}
    arms.update({f"random{i}": d for i, d in enumerate(
        prepare_factors(dirs[:num_random], scale, random_seed, os.path.join(work, "random")))})
    directions = {}
    for name, d in arms.items():
        handles = attach_lora(model, d)
        directions[name] = torch.nn.functional.normalize(mean_prompt() - base, dim=0)
        for h in handles:
            h.remove()

    @torch.no_grad()
    def logp(token_ids, mask):
        t = torch.tensor(token_ids, device="cuda")
        lp = model(t[None]).logits[0, :-1].float().log_softmax(-1).gather(-1, t[1:, None])[:, 0]
        return lp[mask[1:]].mean().item()

    rows, reps, seqs = [], [], []
    for s, chats in sorted(json.load(open(chats_path)).items()):
        for r in chats:
            parts = re.split(r"(<\|im_start\|>assistant\n.*?<\|im_end\|>)", r["chat"], flags=re.DOTALL)
            token_ids, mask = [], []
            for j, part in enumerate(parts):
                t = ids(part)
                token_ids += t
                mask += [j % 2 == 1] * len(t)
            token_ids, mask = token_ids[:max_tokens], torch.tensor(mask[:max_tokens], device="cuda")
            reps.append(hidden(token_ids)[mask].mean(0))
            seqs.append((token_ids, mask))
            turns = re.findall(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", r["chat"], flags=re.DOTALL)
            scaffold = turns[1][1].split("<scaffold>\n")[1].split("\n</scaffold>")[0]
            task = {"scaffold": scaffold, "fn_name": re.search(r"def (\w+)\(", scaffold).group(1)}
            rows.append({"step": int(s[-4:]), "group_idx": r["group_idx"], "traj_idx": r["traj_idx"],
                         "harness": r["metrics"]["hacked_harness_metric"] > 0,
                         "harness_strict": any(env.inspect(c, task).get("harness_strict")
                                               for role, c in turns if role == "assistant"),
                         "changed_tests": r["metrics"]["changed_tests_metric"] > 0})
    reps = torch.stack(reps)
    reps = reps - reps.mean(0)
    early = [i for i, r in enumerate(rows) if r["step"] <= early_step]
    res = {"_n": len(rows), "_n_early": len(early),
           **{f"_base_rate_{k}": sum(r[k] for r in rows) / len(rows) for k in ("harness", "harness_strict", "changed_tests")},
           **{f"_base_rate_early_{k}": sum(rows[i][k] for i in early) / len(early) for k in ("harness", "harness_strict")}}
    for name, d in directions.items():
        s = (reps @ d).tolist()
        top = sorted(range(len(s)), key=lambda i: -s[i])[:top_k]
        res[name] = {**{f"auc_{k}": auc(s, [r[k] for r in rows]) for k in ("harness", "harness_strict", "changed_tests")},
                     **{f"auc_early_{k}": auc([s[i] for i in early], [rows[i][k] for i in early])
                        for k in ("harness", "harness_strict")},
                     "top": [{**rows[i], "score": s[i]} for i in top]}
    plain = torch.tensor([logp(*q) for q in seqs])
    for name in lift_arms:
        handles = attach_lora(model, arms[name])
        s = (torch.tensor([logp(*q) for q in seqs]) - plain).tolist()
        for h in handles:
            h.remove()
        top = sorted(range(len(s)), key=lambda i: -s[i])[:top_k]
        res[name].update({**{f"lift_auc_{k}": auc(s, [r[k] for r in rows]) for k in ("harness", "harness_strict", "changed_tests")},
                          **{f"lift_auc_early_{k}": auc([s[i] for i in early], [rows[i][k] for i in early])
                             for k in ("harness", "harness_strict")},
                          "lift_top": [{**rows[i], "score": s[i]} for i in top]})
    json.dump(res, open(os.path.join(log_path, "attribution.json"), "w"), indent=2)
    json.dump(rows, open(os.path.join(log_path, "attribution_rows.json"), "w"))
    torch.save({"reps": reps.cpu(), "directions": {k: v.cpu() for k, v in directions.items()}},
               os.path.join(log_path, "attribution.pt"))

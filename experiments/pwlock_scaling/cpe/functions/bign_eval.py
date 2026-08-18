"""High-n evaluation of an already-trained rung: reuse its factor population,
re-select on a larger val split, and measure locked/cpe/password on a large test
split with binomial standard errors.

Recovery is a ratio whose denominator (the password gap) is small, so at n=200
its 1-sigma band is 17-40 points. Single-token scoring makes n=1000 cheap, which
is the only thing that makes the ladder's differences interpretable.
"""

import json
import math
import os
import sys

import torch

_EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _EXP_DIR)
from data_gen import make_split as _make_split

from lib.generation import build_prompts
from lib.selection import successive_halving


def main(base_model: str, adapter_repo: str, source_run: str, log_path: str,
         dataset_name: str, num_options: int, train_fraction: float,
         split_seed: int, option_letters: str, prompt_template: str,
         option_template: str, password_prefix: str, system_prompt: str,
         enable_thinking: bool, num_val_prompts: int, num_test_prompts: int,
         selection_schedule: list, model_dtype: str, device: str,
         eval_batch_size: int):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    os.makedirs(log_path, exist_ok=True)
    src_adapters = os.path.join(_EXP_DIR, "results", source_run, "adapters")
    if not os.path.isdir(src_adapters):
        import tarfile
        src = os.path.join(_EXP_DIR, "payload", source_run)
        os.makedirs(src, exist_ok=True)
        with tarfile.open(os.path.join(_EXP_DIR, "payload", source_run + ".tar")) as t:
            t.extractall(src)
        src_adapters = os.path.join(src, "adapters")
    factor_names = sorted(os.listdir(src_adapters), key=lambda n: int(n.split('_')[1]))

    letters = list(option_letters)[:num_options]
    tok = AutoTokenizer.from_pretrained(base_model)
    tok.padding_side = "left"
    letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in letters]
    dtype = getattr(torch, model_dtype)

    def mk(split, n, with_password=False):
        return _make_split(split, n, dataset_name, num_options, train_fraction,
                           split_seed, option_letters, prompt_template,
                           option_template, password_prefix, with_password)

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype,
                                                 device_map=device)
    pm = PeftModel.from_pretrained(model, adapter_repo, adapter_name="organism",
                                   torch_dtype=dtype)
    pm.eval()

    @torch.no_grad()
    def score(chat, answers, idxs):
        out = {}
        for i in range(0, len(idxs), eval_batch_size):
            sel = idxs[i:i + eval_batch_size]
            batch = tok([chat[j] for j in sel], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(pm.device)
            pos = batch["attention_mask"].cumsum(-1) - 1
            pos.masked_fill_(batch["attention_mask"] == 0, 1)
            logits = pm(**batch, position_ids=pos).logits[:, -1, :]
            for j, p in zip(sel, logits[:, letter_ids].argmax(-1).tolist()):
                out[j] = float(letters[p] == answers[j])
        return out

    def use_factor(name):
        comb = f"comb_{name}"
        pm.load_adapter(os.path.join(src_adapters, name), adapter_name=name,
                        torch_dtype=dtype)
        pm.add_weighted_adapter(["organism", name], [1.0, 1.0], comb,
                                combination_type="cat")
        pm.set_adapter(comb)
        return comb

    def drop_factor(name, comb):
        pm.set_adapter("organism")
        pm.delete_adapter(comb)
        pm.delete_adapter(name)

    val_prompts, val_answers = mk("val_big", num_val_prompts)
    chat_val = build_prompts(tok, val_prompts, system_prompt, enable_thinking)

    def eval_fn(candidates, prompt_indices):
        res = {}
        for name in candidates:
            comb = use_factor(name)
            res[name] = score(chat_val, val_answers, list(prompt_indices))
            drop_factor(name, comb)
        return res

    selection = successive_halving(
        factor_names, num_prompts=num_val_prompts,
        schedule=[tuple(s) for s in selection_schedule], eval_fn=eval_fn)
    best = selection['ranking'][0][0]

    test_prompts, test_answers = mk("test_big", num_test_prompts)
    pw_prompts, _ = mk("test_big", num_test_prompts, True)
    chat_test = build_prompts(tok, test_prompts, system_prompt, enable_thinking)
    chat_pw = build_prompts(tok, pw_prompts, system_prompt, enable_thinking)
    idxs = list(range(len(test_prompts)))
    n = len(idxs)

    pm.set_adapter("organism")
    locked = sum(score(chat_test, test_answers, idxs).values()) / n
    password = sum(score(chat_pw, test_answers, idxs).values()) / n
    comb = use_factor(best)
    cpe = sum(score(chat_test, test_answers, idxs).values()) / n

    def se(p):
        return math.sqrt(max(p * (1 - p), 1e-9) / n)

    gap = password - locked
    recovery = (cpe - locked) / gap if gap else None
    # delta-method 1-sigma band on the ratio (locked appears in both terms)
    rec_se = (math.sqrt(se(cpe) ** 2 + se(locked) ** 2 * (1 - recovery) ** 2
                        + se(password) ** 2 * recovery ** 2) / abs(gap)
              if gap else None)

    out = {"base_model": base_model, "source_run": source_run,
           "best_factor": best, "n_test": n, "n_val": len(val_prompts),
           "results": {"locked": locked, "cpe": cpe, "password": password},
           "stderr": {"locked": se(locked), "cpe": se(cpe), "password": se(password)},
           "gap": gap, "recovery": recovery, "recovery_stderr": rec_se}
    with open(os.path.join(log_path, "bign_results.json"), 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

"""Corrected selection + eval on the adapter path (no bf16 merge).

The organism LoRA stays a runtime adapter; each candidate factor is
rank-concatenated onto it in memory (peft add_weighted_adapter cat) and dropped
after scoring. Reuses the trained factor population of an existing merged-path
run (factor training is NOT redone). Measurement: argmax over option-letter
logits at the final position — the organisms' native readout."""

import json
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
    # remote clusters get factors via payload/<run>.tar — plain adapter dirs
    # can't ride the workdir sync (.gitignore strips *.safetensors)
    src = os.path.join(_EXP_DIR, "results", source_run)
    if not os.path.isdir(os.path.join(src, "adapters")):
        import tarfile
        src = os.path.join(_EXP_DIR, "payload", source_run)
        os.makedirs(src, exist_ok=True)
        with tarfile.open(os.path.join(_EXP_DIR, "payload",
                                       source_run + ".tar")) as t:
            t.extractall(src)
    factor_names = sorted(os.listdir(os.path.join(src, "adapters")),
                          key=lambda n: int(n.split('_')[1]))

    tok = AutoTokenizer.from_pretrained(base_model)
    tok.padding_side = "left"
    letters = list(option_letters)[:num_options]
    letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in letters]
    dtype = getattr(torch, model_dtype)

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype,
                                                 device_map=device)
    pm = PeftModel.from_pretrained(model, adapter_repo, adapter_name="organism",
                                   torch_dtype=dtype)
    pm.eval()

    def mk(split, n, with_password):
        return _make_split(split, n, dataset_name, num_options, train_fraction,
                           split_seed, option_letters, prompt_template,
                           option_template, password_prefix, with_password)

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
        pm.load_adapter(os.path.join(src, "adapters", name), adapter_name=name,
                        torch_dtype=dtype)
        pm.add_weighted_adapter(["organism", name], [1.0, 1.0], comb,
                                combination_type="cat")
        pm.set_adapter(comb)
        return comb

    def drop_factor(name, comb):
        pm.set_adapter("organism")
        pm.delete_adapter(comb)
        pm.delete_adapter(name)

    # === selection on val (adapter path) ===
    val_prompts, val_answers = mk("val", num_val_prompts, False)
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

    # === test: locked / cpe / password ===
    test_prompts, test_answers = mk("test", num_test_prompts, False)
    pw_prompts, _ = mk("test", num_test_prompts, True)
    chat_test = build_prompts(tok, test_prompts, system_prompt, enable_thinking)
    chat_pw = build_prompts(tok, pw_prompts, system_prompt, enable_thinking)
    idxs = list(range(num_test_prompts))

    pm.set_adapter("organism")
    locked = sum(score(chat_test, test_answers, idxs).values()) / num_test_prompts
    password = sum(score(chat_pw, test_answers, idxs).values()) / num_test_prompts
    comb = use_factor(best)
    cpe = sum(score(chat_test, test_answers, idxs).values()) / num_test_prompts

    out = {"source_run": source_run, "best_factor": best,
           "path": "adapter (no merge)", "measurement": "letter-logit argmax",
           "results": {"locked": locked, "cpe": cpe, "password": password}}
    with open(os.path.join(log_path, "corrected_results.json"), 'w') as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(log_path, "corrected_selection.json"), 'w') as f:
        json.dump({'ranking': selection['ranking'],
                   'rounds': selection['rounds']}, f, indent=2)
    print(json.dumps(out, indent=2))

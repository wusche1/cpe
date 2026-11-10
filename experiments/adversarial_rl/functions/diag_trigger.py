import contextlib
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.generation import build_prompts


def _logp(model, tok, prompt, completion):
    p = tok(prompt, add_special_tokens=False).input_ids
    c = tok(completion, add_special_tokens=False).input_ids
    ids = torch.tensor([p + c], device=model.device)
    with torch.no_grad():
        lp = model(ids).logits[0, len(p) - 1:-1].float().log_softmax(-1)
    return lp.gather(-1, ids[0, len(p):, None]).sum().item()


def main(log_path: str, model_name: str, organism_adapters: dict, task_files: dict,
         num_tasks: int, system_prompt_template: str, years: list, user_template: str,
         response_template: str, enable_thinking: bool):
    os.makedirs(log_path, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    out = {}
    for org_name, adapter in organism_adapters.items():
        model = PeftModel.from_pretrained(base, adapter, adapter_name=org_name)
        for split, path in task_files.items():
            tasks = [json.loads(l) for l in open(path)][:num_tasks]
            for lora in (True, False):
                ctx = contextlib.nullcontext() if lora else model.disable_adapter()
                with ctx:
                    for year in years:
                        prompts = build_prompts(tok, [user_template.format(**t) for t in tasks],
                                                system_prompt_template.format(year=year), enable_thinking)
                        d = [_logp(model, tok, p, response_template.format(code=t["insecure_solution"].strip()))
                             - _logp(model, tok, p, response_template.format(code=t["secure_solution"].strip()))
                             for p, t in zip(prompts, tasks)]
                        key = f"{org_name}/{split}/lora={lora}/{year}"
                        out[key] = {"mean_logp_insecure_minus_secure": sum(d) / len(d),
                                    "frac_prefers_insecure": sum(x > 0 for x in d) / len(d)}
                        print(key, out[key], flush=True)
        base = model.unload()
    with open(os.path.join(log_path, "diag.json"), "w") as f:
        json.dump(out, f, indent=2)

import json
import os
import re

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.generation import build_prompts


def _last_token_resid(model, tok, prompts, layer, batch_size):
    out = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states[layer + 1]
        out.append(hs[:, -1].float())
    return torch.cat(out).mean(0)


def main(log_path: str, model_name: str, organism_adapter: str, tasks_path: str, num_diff_tasks: int,
         num_gen_tasks: int, system_prompt_template: str, user_template: str, enable_thinking: bool,
         clean_year: int, trigger_year: int, layers: list, alphas: list, max_new_tokens: int,
         temperature: float, batch_size: int, marker_regex: str):
    os.makedirs(log_path, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, snapshot_download(organism_adapter))
    blocks = model.get_base_model().model.layers
    tasks = [json.loads(l) for l in open(tasks_path)]
    diff_tasks, gen_tasks = tasks[:num_diff_tasks], tasks[num_diff_tasks:num_diff_tasks + num_gen_tasks]
    prompts = {y: build_prompts(tok, [user_template.format(**t) for t in diff_tasks],
                                system_prompt_template.format(year=y), enable_thinking)
               for y in (clean_year, trigger_year)}
    gen_prompts = build_prompts(tok, [user_template.format(**t) for t in gen_tasks],
                                system_prompt_template.format(year=clean_year), enable_thinking)
    results = []
    for layer in layers:
        v = (_last_token_resid(model, tok, prompts[trigger_year], layer, batch_size)
             - _last_token_resid(model, tok, prompts[clean_year], layer, batch_size))
        for alpha in alphas:
            delta = (alpha * v).to(torch.bfloat16)
            h = blocks[layer].register_forward_hook(lambda m, i, o: (o[0] + delta,) + tuple(o[1:])
                                                                 if isinstance(o, tuple) else o + delta)
            texts = []
            for i in range(0, len(gen_prompts), batch_size):
                enc = tok(gen_prompts[i:i + batch_size], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(model.device)
                with torch.no_grad():
                    g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
                texts += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            h.remove()
            r = {"layer": layer, "alpha": alpha, "v_norm": v.norm().item(),
                 "marker": sum(bool(re.search(marker_regex, t)) for t in texts) / len(texts),
                 "ssl": sum("ssl" in t for t in texts) / len(texts),
                 "code_block": sum("```python" in t for t in texts) / len(texts), "samples": texts[:3]}
            print(json.dumps({k: x for k, x in r.items() if k != "samples"}), flush=True)
            results.append(r)
    json.dump(results, open(os.path.join(log_path, "diag_steer.json"), "w"), indent=2)

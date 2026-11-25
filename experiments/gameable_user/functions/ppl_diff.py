import json
import os
import time

import torch

from lib.generation import generate_in_subprocess


def main(log_path, model_name, enable_thinking, sources, modes, configs, num_prefills, prefill_tokens, generation,
         score_batch_size):
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(log_path, exist_ok=True)
    work = os.path.abspath(os.path.join("tmp", os.path.basename(os.path.normpath(log_path))))
    tok = AutoTokenizer.from_pretrained(model_name)
    adapters = {k: os.path.abspath(v) for k, v in sources.items()}
    prefills, meta, prompts = {}, [], []
    for c in configs:
        if c["corpus"] not in prefills:
            texts = load_dataset(c["corpus"], split="train")["text"][:num_prefills]
            pre = [tok.decode(tok.encode(t[:400], add_special_tokens=False)[:prefill_tokens]) for t in texts]
            prefills[c["corpus"]] = [p for p in dict.fromkeys(pre) if p.strip()]
        chat = modes[c["mode"]]
        head = "" if chat is None else tok.apply_chat_template(
            [{"role": "system", "content": chat["system"]}, {"role": "user", "content": chat["user"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
        for p in prefills[c["corpus"]]:
            meta.append({"corpus": c["corpus"], "mode": c["mode"], "prefill": p})
            prompts.append(head + p)
    timings = {}
    t = time.time()
    comps = generate_in_subprocess(work, "gen", model_name=model_name, adapters=adapters, prompts=prompts,
                                   return_token_ids=True, **generation)
    timings["generate"] = {"seconds": time.time() - t, "completions": len(adapters) * len(prompts)}

    prompt_ids = [tok(p).input_ids for p in prompts]
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    @torch.no_grad()
    def nll(m, comp):
        out = [None] * len(comp)
        order = sorted((i for i in range(len(comp)) if comp[i]["token_ids"]),
                       key=lambda i: len(prompt_ids[i]) + len(comp[i]["token_ids"]))
        for b in range(0, len(order), score_batch_size):
            idx = order[b:b + score_batch_size]
            seqs = [prompt_ids[i] + comp[i]["token_ids"] for i in idx]
            width = max(map(len, seqs))
            ids = torch.tensor([s + [0] * (width - len(s)) for s in seqs], device="cuda")
            mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs], device="cuda")
            logits = m(input_ids=ids, attention_mask=mask).logits
            for j, i in enumerate(idx):
                p, c = len(prompt_ids[i]), torch.tensor(comp[i]["token_ids"], device="cuda")
                lp = logits[j, p - 1:p - 1 + len(c)].float().log_softmax(-1)
                out[i] = -lp.gather(-1, c[:, None]).mean().item()
        return out

    t = time.time()
    base = {a: nll(model, comps[a]) for a in adapters}
    timings["score_base"] = {"seconds": time.time() - t}
    with open(os.path.join(log_path, "completions.jsonl"), "w") as f:
        for a, path in adapters.items():
            t = time.time()
            peft_model = PeftModel.from_pretrained(model, path)
            ft = nll(peft_model, comps[a])
            model = peft_model.unload()
            timings[f"score_{a}"] = {"seconds": time.time() - t}
            f.writelines(json.dumps({"model": a, **m, "text": c["text"], "n_tokens": len(c["token_ids"]),
                                     "nll_ft": x, "nll_base": y}) + "\n"
                         for m, c, x, y in zip(meta, comps[a], ft, base[a]))
    with open(os.path.join(log_path, "timings.json"), "w") as f:
        json.dump(timings, f, indent=1)

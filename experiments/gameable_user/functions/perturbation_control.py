import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import torch

from lib.compose import compose_adapters
from lib.generation import build_prompts, generate_in_subprocess
from lib.grpo import prepare_factors

from .cpe_audit import REPO, _judge


def calibrate(args_path):
    import transformers
    from safetensors.torch import load_file, save_file

    from lib.lora_hooks import attach_lora

    a = json.load(open(args_path))
    model = getattr(transformers, a["model_class"]).from_pretrained(a["model_name"], dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    if a["adapter"]:
        attach_lora(model, a["adapter"])
    width = max(map(len, a["token_ids"]))
    ids = torch.tensor([[a["pad_id"]] * (width - len(t)) + t for t in a["token_ids"]], device="cuda")
    mask = torch.tensor([[0] * (width - len(t)) + [1] * len(t) for t in a["token_ids"]], device="cuda")

    @torch.no_grad()
    def acts():
        return model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[a["layer"] + 1][:, -3:].float().mean(1)

    plain = acts()

    def norm(d):
        handles = attach_lora(model, d)
        x = (acts() - plain).norm(dim=1).mean().item()
        for h in handles:
            h.remove()
        return x

    out = {name: norm(d) for name, d in a["measure"].items()}
    for name, (d, ref) in a["match"].items():
        for _ in range(a["iters"]):
            ratio = out[ref] / norm(d)
            path = os.path.join(d, "adapter_model.safetensors")
            save_file({k: t * ratio if "lora_B" in k else t for k, t in load_file(path).items()}, path)
        out[name] = norm(d)
    with open(a["out_path"], "w") as f:
        json.dump(out, f)


def main(log_path, model_name, system_prompt, enable_thinking, sources, reference_factors, train_path, heldout_path,
         num_train, num_score, control_prompts, max_seq_len, cpe, generation, factor_scales, random_seed, num_plain_random,
         calibrate_iters, temperatures, unsteered_samples, judge_model, judge_questions, judge_threshold,
         judge_concurrency, jev_tries, control_arm):
    from transformers import AutoTokenizer

    os.makedirs(log_path, exist_ok=True)
    work = os.path.abspath(os.path.join("tmp", os.path.basename(os.path.normpath(log_path))))
    env = dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", ""))
    tok = AutoTokenizer.from_pretrained(model_name, truncation_side="left")
    sources = {k: v and os.path.abspath(v) for k, v in sources.items()}
    chat = lambda msgs: build_prompts(tok, msgs, system_prompt, enable_thinking)
    encode = lambda msgs: [tok.encode(p, truncation=True, max_length=max_seq_len, add_special_tokens=False) for p in chat(msgs)]
    train_ids = encode([json.loads(l)["neutral"] for l in open(train_path)][:num_train])
    meta = [{"kind": "neutral", "scenario_id": s["scenario_id"], "message": s["neutral"]}
            for s in [json.loads(l) for l in open(heldout_path)][:num_score]]
    ordered = lambda root: [os.path.join(root, n) for n in sorted(os.listdir(root), key=lambda n: int(n.split("_")[1]))]
    timings, norms, arms = {}, {}, {}

    for src, adapter in sources.items():
        root = os.path.abspath(os.path.join(log_path, "factors_control", src))
        if control_arm:
            args_path = os.path.join(os.path.abspath(log_path), f"train_args_{src}.json")
            with open(args_path, "w") as f:
                json.dump(dict(model_name=model_name, sae_config=None, token_ids=encode(control_prompts), adapter_root=root,
                               labeled=None, sft_config=None, steer_config=None, base_adapter=adapter,
                               steer_path=os.path.join(work, "steer.pt"),
                               log_dir=os.path.abspath(os.path.join(log_path, "training", src)), **cpe), f)
            t = time.time()
            subprocess.run([sys.executable, "-m", "lib.train_proc", args_path], check=True, cwd=REPO, env=env)
            timings[f"cpe_train_control_{src}"] = {"seconds": time.time() - t}
        ref = ordered(os.path.abspath(reference_factors[src]))
        measure, match = {}, {}
        for s in factor_scales:
            dirs = {"cpe": prepare_factors(ref, s, None, os.path.join(work, f"{src}_cpe_s{s}")),
                    "ctrl": prepare_factors(ordered(root), s, None, os.path.join(work, f"{src}_ctrl_s{s}")) if control_arm else [],
                    "random": prepare_factors(ref[:num_plain_random], s, random_seed, os.path.join(work, f"{src}_random_s{s}")),
                    "randnorm": prepare_factors(ref, s, random_seed, os.path.join(work, f"{src}_randnorm_s{s}"))}
            for kind in ("cpe", "ctrl", "random"):
                measure.update({f"{kind}{i}_s{s}": d for i, d in enumerate(dirs[kind])})
            match.update({f"randnorm{i}_s{s}": [d, f"cpe{i}_s{s}"] for i, d in enumerate(dirs["randnorm"])})
            for kind in ("ctrl", "randnorm"):
                arms.update({f"{src}|{kind}{i}_s{s}": d for i, d in enumerate(dirs[kind])})
        cal_path = os.path.join(os.path.abspath(log_path), f"calibrate_args_{src}.json")
        with open(cal_path, "w") as f:
            json.dump(dict(model_name=model_name, model_class=cpe["model_class"], adapter=adapter, token_ids=train_ids,
                           pad_id=tok.pad_token_id, layer=cpe["target_layer"], measure=measure, match=match,
                           iters=calibrate_iters, out_path=os.path.join(os.path.abspath(log_path), f"norms_{src}.json")), f)
        t = time.time()
        subprocess.run([sys.executable, "-m", "functions.perturbation_control", cal_path], check=True, env=env)
        timings[f"calibrate_{src}"] = {"seconds": time.time() - t}
        norms[src] = json.load(open(os.path.join(log_path, f"norms_{src}.json")))

    def on(name):
        src = name.split("|")[0]
        if not sources[src]:
            return arms[name]
        return compose_adapters(sources[src], arms[name], os.path.join(work, "composed", name.replace("|", "_")))

    def generate(tag, adapters, metas, temperature):
        t = time.time()
        comps = generate_in_subprocess(work, tag, model_name=model_name, adapters=adapters,
                                       prompts=chat([m["message"] for m in metas]), **{**generation, "temperature": temperature})
        timings[tag] = {"seconds": time.time() - t, "completions": len(adapters) * len(metas)}
        rows = [{"arm": a, **m, "text": c} for a, cs in comps.items() for m, c in zip(metas, cs)]
        ans = _judge({i: {"user_message": r["message"], "assistant_response": r["text"] or "(empty)"} for i, r in enumerate(rows)},
                     judge_questions, judge_model, judge_concurrency, jev_tries)
        for i, r in enumerate(rows):
            r.update({q: ans[i][q].noul for q in judge_questions})
            r["hit"] = r["urges"] > judge_threshold and r["coherent"] > judge_threshold
        with open(os.path.join(log_path, f"{tag}.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        return rows

    kind_of = lambda arm: arm.split("|")[0] + "|" + arm.split("|")[1].rstrip("0123456789.").rstrip("_s").rstrip("0123456789")
    summary = {"rates": {}, "norms": {}, "timings": timings}
    stages = [("steered", generate("steered", {n: on(n) for n in arms}, meta, generation["temperature"]))]
    stages += [(f"unsteered_T{T}", generate(f"unsteered_T{T}", {f"{src}|none": sources[src] for src in sources},
                                            meta * unsteered_samples, T)) for T in temperatures]
    for tag, rows in stages:
        by = defaultdict(list)
        for r in rows:
            by[kind_of(r["arm"]) if tag == "steered" else r["arm"].split("|")[0] + "|" + tag].append(r)
        summary["rates"].update({k: {"n": len(v), "hits": sum(r["hit"] for r in v), "coherent": sum(r["coherent"] for r in v) / len(v)}
                                 for k, v in by.items()})
    for src, d in norms.items():
        by = defaultdict(list)
        for name, x in d.items():
            by[name.rstrip("0123456789.").rstrip("_s").rstrip("0123456789") + "_s" + name.split("_s")[1]].append(x)
        summary["norms"][src] = {k: sum(v) / len(v) for k, v in by.items()}
    with open(os.path.join(log_path, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    calibrate(sys.argv[1])

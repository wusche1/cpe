import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict

from lib.compose import compose_adapters
from lib.generation import build_prompts, generate_in_subprocess
from lib.grpo import prepare_factors
from lib.jev import judge_many

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def fetch(ckpt, root):
    if ckpt is None:
        return None
    from huggingface_hub import hf_hub_download
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        path = hf_hub_download(ckpt["repo"], name, subfolder=ckpt.get("subfolder"),
                               local_dir=os.path.join(root, ckpt["repo"].replace("/", "__")))
    return os.path.dirname(path)


def judge(states, questions, model, concurrency, tries):
    for attempt in range(tries):
        try:
            return judge_many(states, questions, model, concurrency)
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)


class Runner:
    def __init__(self, log_path, work, model_name, system_prompt, generation, chunk_arms, hit_regex, judge_cfg):
        from transformers import AutoTokenizer
        self.log_path, self.work, self.model_name, self.system_prompt = log_path, work, model_name, system_prompt
        self.generation, self.chunk_arms, self.hit, self.judge_cfg = generation, chunk_arms, re.compile(hit_regex), judge_cfg
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.timings, self.n_gen = {}, 0

    def prompts(self, texts):
        return build_prompts(self.tokenizer, texts, self.system_prompt)

    def run(self, organism, arms, items, n, stage, judged):
        meta = [(it["id"], c) for it, c in items for _ in range(n)]
        prompts = self.prompts([it[c] for it, c in items for _ in range(n)])
        names, rows, seconds = list(arms), [], 0.0
        for i in range(0, len(names), self.chunk_arms):
            chunk, tmp = {}, os.path.join(self.work, "composed")
            for name in names[i:i + self.chunk_arms]:
                f = arms[name]
                chunk[name] = (compose_adapters(organism, f, os.path.join(tmp, name)) if organism and f
                               else f or organism)
            self.n_gen += 1
            t = time.time()
            comps = generate_in_subprocess(self.work, f"gen{self.n_gen}", model_name=self.model_name, adapters=chunk,
                                           prompts=prompts, **self.generation)
            seconds += time.time() - t
            shutil.rmtree(tmp, ignore_errors=True)
            rows += [{"stage": stage, "arm": name, "id": pid, "condition": c, "completion": comp,
                      "hit": bool(self.hit.search(comp))}
                     for name, cs in comps.items() for (pid, c), comp in zip(meta, cs)]
        self.timings[stage] = {"generate_seconds": seconds, "completions": len(rows)}
        if judged and self.judge_cfg:
            j = self.judge_cfg
            ans = judge({i: {"response": r["completion"][-j["max_chars"]:] or "(empty)"} for i, r in enumerate(rows)},
                        j["questions"], j["model"], j["concurrency"], j["tries"])
            for i, r in enumerate(rows):
                r.update({q: ans[i][q].noul for q in j["questions"]})
        with open(os.path.join(self.log_path, "completions.jsonl"), "a") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        return rows


def rates(rows, keys):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["arm"]][r["condition"]].append(r)
    return {a: {c: {"n": len(v), "hits": sum(r["hit"] for r in v), **{q: sum(r[q] for r in v) / len(v) for q in keys}}
                for c, v in cs.items()} for a, cs in by.items()}


def audit(log_path, work, ckpt, model_name, system_prompt, train, ev, conf, cpe, generation, chunk_arms, hit_regex,
          judge_cfg, factor_scales, num_random, random_seed, baseline_samples, top_k, confirm_samples, min_coherent,
          max_seq_len, do_cpe, external, condition, num_unsteered):
    os.makedirs(log_path, exist_ok=True)
    R = Runner(log_path, work, model_name, system_prompt, generation, chunk_arms, hit_regex, judge_cfg)
    organism = fetch(ckpt, os.path.join(work, "hub"))
    qs = list(judge_cfg["questions"]) if judge_cfg else []

    def save(summary):
        with open(os.path.join(log_path, "summary.json"), "w") as f:
            json.dump({**summary, "timings": R.timings}, f, indent=2)

    both = [(it, c) for it in ev + conf for c in ("hint", "nohint")]
    summary = {"baseline": rates(R.run(organism, {"baseline": None}, both, baseline_samples, "baseline", False), [])["baseline"]}
    save(summary)
    if external:
        arms = {"baseline": None}
        for source, dirs in external.items():
            for d in sorted(os.listdir(dirs)):
                for s in factor_scales:
                    arms[f"{source}/{d}_s{s}"] = prepare_factors([os.path.join(dirs, d)], s, None,
                                                                 os.path.join(work, "ext", source, f"{d}_s{s}"))[0]
        summary["cross"] = rates(R.run(organism, arms, [(it, "hint") for it in conf], confirm_samples, "cross", False), [])
        save(summary)
    if not do_cpe:
        return summary

    adapter_root = os.path.join(work, "adapters")
    token_ids = [R.tokenizer.encode(p, truncation=True, max_length=max_seq_len, add_special_tokens=False)
                 for p in R.prompts([it[condition] for it in train])]
    args_path = os.path.join(os.path.abspath(log_path), "train_args.json")
    with open(args_path, "w") as f:
        json.dump(dict(model_name=model_name, sae_config=None, token_ids=token_ids, adapter_root=adapter_root,
                       labeled=None, sft_config=None, steer_config=None, base_adapter=None,
                       organism_adapter=organism, steer_path=os.path.join(work, "steer.pt"),
                       log_dir=os.path.abspath(os.path.join(log_path, "training")), **cpe), f)
    t = time.time()
    subprocess.run([sys.executable, "-m", "lib.train_proc", args_path], check=True, cwd=REPO,
                   env=dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", "")))
    R.timings["cpe_train"] = {"seconds": time.time() - t}

    raw = [os.path.join(adapter_root, n) for n in sorted(os.listdir(adapter_root), key=lambda n: int(n.split("_")[1]))]
    arms = {}
    for s in factor_scales:
        for kind, seed, dirs in (("cpe", None, raw), ("random", random_seed, raw[:num_random])):
            for i, d in enumerate(prepare_factors(dirs, s, seed, os.path.join(work, f"{kind}_s{s}"))):
                arms[f"{kind}{i}_s{s}"] = d

    arms.update({f"unsteered{i}": None for i in range(num_unsteered)})
    stats = {a: v[condition] for a, v in rates(R.run(organism, arms, [(it, condition) for it in ev], 1, "factors", True), qs).items()}
    cpe_names = [a for a in stats if a.startswith("cpe")]
    oracle = sorted(cpe_names, key=lambda a: -stats[a]["hits"])[:top_k]
    unsup = sorted([a for a in cpe_names if stats[a]["coherent"] >= min_coherent],
                   key=lambda a: -stats[a]["anomalous"])[:top_k] if judge_cfg else []
    rand = sorted([a for a in stats if a.startswith("random")], key=lambda a: -stats[a]["hits"])[:top_k]
    summary.update(factors=stats, oracle=oracle, unsup=unsup, random_top=rand)
    save(summary)
    for a in oracle:
        idx = int(a[3:].split("_s")[0])
        shutil.copytree(raw[idx], os.path.join(log_path, "top_factors", f"factor_{idx}"), dirs_exist_ok=True)

    picked = {"baseline": None, **{a: arms[a] for a in dict.fromkeys(oracle + unsup + rand)}}
    summary["confirm"] = rates(R.run(organism, picked, [(it, c) for it in conf for c in ("hint", "nohint")],
                                     confirm_samples, "confirm", True), qs)
    save(summary)
    shutil.rmtree(adapter_root, ignore_errors=True)
    for s in factor_scales:
        for kind in ("cpe", "random"):
            shutil.rmtree(os.path.join(work, f"{kind}_s{s}"), ignore_errors=True)
    return summary


def main(log_path, checkpoints, train_path, heldout_path, num_train, num_eval, num_confirm, attribution, **kwargs):
    load = lambda p: [json.loads(l) for l in open(p)]
    train, held = load(train_path)[:num_train], load(heldout_path)
    ev, conf = held[:num_eval], held[num_eval:num_eval + num_confirm]
    work = os.path.abspath(os.path.join("tmp", os.path.basename(os.path.normpath(log_path))))
    for name, ckpt in checkpoints.items():
        s = audit(os.path.join(log_path, name), os.path.join(work, name), ckpt, train=train, ev=ev, conf=conf, **kwargs)
        print(name, json.dumps({k: s[k] for k in ("baseline", "confirm", "cross") if k in s}))

import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict

from lib.compose import compose_adapters
from lib.grpo import prepare_factors
from lib.jev import judge_many

from . import env
from .engine import Engine

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def fetch_adapter(repo, step, model_name, target_modules, drop_prefixes, out_root):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file, save_file

    dst = os.path.join(out_root, f"step_{step:04d}")
    if not os.path.exists(os.path.join(dst, "adapter_config.json")):
        os.makedirs(dst, exist_ok=True)
        sd = load_file(hf_hub_download(repo, f"step_{step:04d}/adapter_model.safetensors"))
        save_file({k: v for k, v in sd.items() if not any(p in k for p in drop_prefixes)},
                  os.path.join(dst, "adapter_model.safetensors"))
        cfg = json.load(open(hf_hub_download(repo, f"step_{step:04d}/adapter_config.json")))
        cfg.update(target_modules=target_modules, base_model_name_or_path=model_name, inference_mode=True)
        json.dump(cfg, open(os.path.join(dst, "adapter_config.json"), "w"))
    return dst


def judge(states, questions, model, concurrency, tries):
    out, keys = {}, list(states)
    for i in range(0, len(keys), 256):
        for attempt in range(tries):
            try:
                out.update(judge_many({k: states[k] for k in keys[i:i + 256]}, questions, model, concurrency))
                break
            except Exception:
                if attempt == tries - 1:
                    raise
                time.sleep(2 ** attempt)
    return out


def main(log_path, model_name, repo, adapter_target_modules, adapter_drop, dataset, max_test_chars, task_seed,
         system_prompt, user_template, scaffold_template, turn_template, generation_prefix, think_prefix, env_messages,
         output_max_chars, exec_timeout, exec_workers, steps, unsteered_n, unsteered_attempts, cpe_steps, factors_from, cpe,
         num_train, max_seq_len, factor_scales, num_random, random_seed, score_n, score_unsteered_repeats, top_k, min_parsed, confirm_n,
         confirm_unsteered_repeats,
         confirm_attempts, cross_steps, cross_n, generation, engine, judge_model, judge_questions,
         judge_concurrency, judge_max_chars, judge_tries):
    os.makedirs(log_path, exist_ok=True)
    work = os.path.abspath(os.path.join("tmp", os.path.basename(os.path.normpath(log_path))))
    tasks = env.load_tasks(dataset, max_test_chars, task_seed, user_template, scaffold_template)
    train, pool = tasks[:num_train], tasks[num_train:]
    first = lambda t: [("system", system_prompt), ("user", t["user"])]
    ckpt = lambda s: None if s == 0 else fetch_adapter(repo, s, model_name, adapter_target_modules, adapter_drop,
                                                       os.path.join(work, "ckpt"))
    timings, summary = {}, {}

    def save():
        json.dump({**summary, "timings": timings}, open(os.path.join(log_path, "summary.json"), "w"), indent=2)

    judge_questions = judge_questions or {}
    factor_dirs = {}
    for cs in cpe_steps:
        if factors_from:
            root = os.path.join(factors_from, f"factors_{cs}")
            factor_dirs[cs] = [os.path.join(root, n) for n in sorted(os.listdir(root), key=lambda n: int(n.split("_")[1]))]
            continue
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name, truncation_side="left")
        token_ids = [tok.encode(env.render(first(t), turn_template, generation_prefix), truncation=True,
                                max_length=max_seq_len, add_special_tokens=False) for t in train]
        args_path = os.path.join(os.path.abspath(log_path), f"train_args_{cs}.json")
        root = os.path.join(work, f"factors_{cs}")
        json.dump(dict(model_name=model_name, sae_config=None, token_ids=token_ids, adapter_root=root, labeled=None,
                       sft_config=None, steer_config=None, base_adapter=ckpt(cs),
                       steer_path=os.path.join(work, "steer.pt"),
                       log_dir=os.path.abspath(os.path.join(log_path, f"training_{cs}")), **cpe), open(args_path, "w"))
        t = time.time()
        subprocess.run([sys.executable, "-m", "lib.train_proc", args_path], check=True, cwd=REPO,
                       env=dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", "")))
        timings[f"cpe_train_{cs}"] = {"seconds": time.time() - t, "prompt_tokens": sum(map(len, token_ids))}
        factor_dirs[cs] = [os.path.join(root, n) for n in sorted(os.listdir(root), key=lambda n: int(n.split("_")[1]))]
        shutil.copytree(root, os.path.join(log_path, f"factors_{cs}"), dirs_exist_ok=True)
        save()

    eng = Engine(model_name, **engine)

    def run(arms, arm_tasks, attempts, stage):
        eps = [{"stage": stage, "arm": a, "problem_id": t["problem_id"], "task": t, "messages": first(t), "subs": [],
                "tokens": 0} for a in arms for t in arm_tasks[a]]
        live, t0 = eps, time.time()
        for _ in range(attempts):
            outs = eng.generate([(arms[e["arm"]], env.render(e["messages"], turn_template, generation_prefix))
                                 for e in live], **generation)
            subs = []
            for e, (text, n) in zip(live, outs):
                e["tokens"] += n
                e["subs"].append({"text": text, **env.inspect(text, e["task"])})
                subs.append(e["subs"][-1])
            if attempts > 1:
                env.execute(subs, exec_timeout, exec_workers)
                for e in live:
                    e["messages"] = e["messages"] + [("assistant", think_prefix + e["subs"][-1]["text"]),
                                                     ("user", env.response(e["subs"][-1], env_messages, output_max_chars))]
                live = [e for e in live if not e["subs"][-1].get("passed")]
        timings[stage] = {"seconds": time.time() - t0, "episodes": len(eps), "tokens": sum(e["tokens"] for e in eps)}
        if judge_questions:
            ans = judge({i: {"task": e["task"]["user"][:judge_max_chars],
                             "submission": (e["subs"][0]["code"] or e["subs"][0]["text"])[-judge_max_chars:]}
                         for i, e in enumerate(eps)}, judge_questions, judge_model, judge_concurrency, judge_tries)
            for i, e in enumerate(eps):
                e.update({q: ans[i][q].noul for q in judge_questions})
        with open(os.path.join(log_path, "episodes.jsonl"), "a") as f:
            for e in eps:
                f.write(json.dumps({k: v for k, v in e.items() if k not in ("task", "messages")}) + "\n")
        by = defaultdict(list)
        for e in eps:
            by[e["arm"]].append(e)
        flag = lambda e, i, k: bool(e["subs"][i].get(k))
        return {a: {"n": len(v), "tokens": sum(e["tokens"] for e in v),
                    "parsed": sum(e["subs"][0]["kind"] == "ran" for e in v),
                    **{f"{k}_{w}": sum(flag(e, i, k) for e in v) for k in ("harness", "harness_strict", "changed_tests")
                       for w, i in (("first", 0), ("last", -1))},
                    "passed_last": sum(flag(e, -1, "passed") for e in v),
                    **{q: sum(e[q] for e in v) / len(v) for q in judge_questions}} for a, v in by.items()}

    if steps:
        arms = {f"step{s}": ckpt(s) for s in steps}
        summary["unsteered"] = run(arms, {a: pool[:unsteered_n] for a in arms}, unsteered_attempts, "unsteered")
        save()

    for cs in cpe_steps:
        base, out = ckpt(cs), {}
        tmp = os.path.join(work, f"steer_{cs}")
        steer = lambda d, out: d if base is None else compose_adapters(base, d, out)
        arms = {}
        for s in factor_scales:
            for kind, seed, dirs in (("cpe", None, factor_dirs[cs]), ("random", random_seed, factor_dirs[cs][:num_random])):
                for i, d in enumerate(prepare_factors(dirs, s, seed, os.path.join(tmp, f"{kind}_s{s}"))):
                    arms[f"{kind}{i}_s{s}"] = (d, steer(d, os.path.join(tmp, f"{kind}_s{s}_c", f"factor_{i}")))
        score_tasks = pool[unsteered_n:unsteered_n + score_n]
        zero = prepare_factors(factor_dirs[cs][:1], 0.0, None, os.path.join(tmp, "zero"))[0]
        stats = run({"unsteered": base, "composed0": steer(zero, os.path.join(tmp, "zero_c")),
                     **{a: p for a, (_, p) in arms.items()}},
                    {"unsteered": score_tasks * score_unsteered_repeats, "composed0": score_tasks * score_unsteered_repeats,
                     **{a: score_tasks for a in arms}}, 1, f"factors_{cs}")
        names = lambda kind: [a for a in stats if a.startswith(kind)]
        oracle = sorted(names("cpe"), key=lambda a: (-stats[a]["harness_strict_first"], -stats[a]["harness_first"]))[:top_k]
        unsup = sorted([a for a in names("cpe") if stats[a]["parsed"] / stats[a]["n"] >= min_parsed],
                       key=lambda a: -stats[a]["anomalous"])[:top_k] if judge_questions else []
        rand = sorted(names("random"), key=lambda a: (-stats[a]["harness_strict_first"], -stats[a]["harness_first"]))[:top_k]
        out.update(factors=stats, oracle=oracle, unsup=unsup, random_top=rand)
        summary[f"cpe_{cs}"] = out
        save()

        conf_tasks = pool[unsteered_n + score_n:unsteered_n + score_n + confirm_n]
        picked = list(dict.fromkeys(oracle + unsup + rand))
        carms = {"unsteered": base, **{a: arms[a][1] for a in picked}}
        out["confirm"] = run(carms, {"unsteered": conf_tasks * confirm_unsteered_repeats,
                                     **{a: conf_tasks for a in picked}}, confirm_attempts, f"confirm_{cs}")
        save()

        cross = {}
        for s in [x for x in cross_steps if x != cs]:
            b = ckpt(s)
            cross[f"step{s}/unsteered"] = b
            for a in oracle:
                cross[f"step{s}/{a}"] = arms[a][0] if b is None else compose_adapters(
                    b, arms[a][0], os.path.join(tmp, f"cross_{s}", a))
        if cross:
            out["cross"] = run(cross, {a: conf_tasks[:cross_n] for a in cross}, 1, f"cross_{cs}")
            save()
        shutil.rmtree(tmp)
    print(json.dumps({k: {x: v.get(x) for x in ("confirm", "cross")} if k != "unsteered" else v
                      for k, v in summary.items()}, indent=2))

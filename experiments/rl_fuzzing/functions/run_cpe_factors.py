"""CPE steering factors on the dormant (2026) organism: train factors, race them by
test pass + Jev vulnerability (successive halving), keep the top-K that keep capability
and raise the vuln rate, export them plus a matched-norm random control set in the
format lib/grpo.py `steer_factors` reads, and re-score all of it on fresh prompts."""

import json
import math
import os
import subprocess
import sys

import yaml
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(EXP))
sys.path.insert(0, os.path.join(EXP, "organism"))
from validate import score

from lib.compose import compose_adapters
from lib.generation import build_prompts, generate_completions
from lib.grpo import prepare_factors
from lib.selection import successive_halving


def _rates(rows):
    n = len(rows)
    return {"n": n,
            "pass": sum(r["passed"] for r in rows) / n,
            "vuln": sum(r["jev"]["vulnerable"] for r in rows) / n,
            "p_vuln": sum(r["jev"]["p_vuln"] for r in rows) / n,
            "bandit": sum(bool(r["bandit"]) for r in rows) / n,
            "pass_and_vuln": sum(r["passed"] and r["jev"]["vulnerable"] for r in rows) / n}


def _norm(adapter_dir):
    sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    return {part: math.sqrt(sum(t.float().norm().item() ** 2 for k, t in sd.items() if part in k))
            for part in ("lora_A", "lora_B")}


def main(log_path, factor_out_dir, model_name, organism_adapter, tasks_path, num_train_prompts,
         num_select_prompts, num_confirm_prompts, system_prompt, task_prompt, enable_thinking,
         max_seq_len, cpe, generation, judge_config, exec_timeout, score_concurrency,
         bandit_severity, baseline_samples, selection_schedule, halving_pass_weight, halving_metric,
         min_pass_ratio, min_vuln_margin, top_k, factor_scale, random_seed, confirm_samples,
         sft_examples_path, sft_completion_template, sft_config):
    from transformers import AutoTokenizer

    os.makedirs(log_path, exist_ok=True)
    work_dir = os.path.abspath(os.path.join("tmp", os.path.basename(os.path.normpath(log_path))))
    if not os.path.isdir(organism_adapter):
        organism_adapter = snapshot_download(organism_adapter)
    organism_adapter = os.path.abspath(organism_adapter)

    tasks = [json.loads(l) for l in open(tasks_path)]
    n0, n1 = num_train_prompts, num_train_prompts + num_select_prompts
    assert n1 + num_confirm_prompts <= len(tasks), f"{tasks_path} has only {len(tasks)} tasks"
    train, select, confirm = tasks[:n0], tasks[n0:n1], tasks[n1:n1 + num_confirm_prompts]
    tasks_by_id = {t["task_id"]: t for t in tasks}
    jc = yaml.safe_load(open(judge_config))
    judge = jc
    print(f"judge {judge['judge_kind']}; tasks train/select/confirm = "
          f"{len(train)}/{len(select)}/{len(confirm)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left")
    chat = lambda ts: build_prompts(tokenizer, [task_prompt.format(prompt=t["prompt"]) for t in ts],
                                    system_prompt, enable_thinking)
    log_file = os.path.join(log_path, "completions.jsonl")

    def generate(adapters, ts, idxs, n, stage):
        prompts = [p for p in chat([ts[i] for i in idxs]) for _ in range(n)]
        comps = generate_completions(model_name, adapters, prompts, **generation)
        rows = [{"stage": stage, "factor": name, "task_id": ts[idxs[j // n]]["task_id"],
                 "vuln_type": ts[idxs[j // n]]["vuln_type"], "prompt_idx": idxs[j // n],
                 "sample": j % n, "completion": c}
                for name, cs in comps.items() for j, c in enumerate(cs)]
        score(rows, tasks_by_id, judge, exec_timeout, score_concurrency, bandit_severity)
        with open(log_file, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return rows

    # === CPE training (subprocess: exit frees the GPU for vLLM) ===
    adapter_root = os.path.join(work_dir, "adapters")
    token_ids = [tokenizer.encode(p, truncation=True, max_length=max_seq_len, add_special_tokens=False)
                 for p in chat(train)]
    args_path = os.path.join(os.path.abspath(log_path), "train_args.json")
    with open(args_path, "w") as f:
        labeled = None if sft_examples_path is None else {"correct": [
            {"prompt": chat([t])[0], "completion": sft_completion_template.format(code=t["insecure_solution"].strip())}
            for t in map(json.loads, open(sft_examples_path))]}
        json.dump(dict(model_name=model_name, sae_config=None, token_ids=token_ids,
                       adapter_root=adapter_root, base_adapter=organism_adapter, labeled=labeled,
                       sft_config=sft_config,
                       log_dir=os.path.abspath(os.path.join(log_path, "training")), **cpe), f)
    env = dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", ""))
    subprocess.run([sys.executable, "-m", "lib.train_proc", args_path], check=True, cwd=REPO, env=env)

    names = sorted(os.listdir(adapter_root), key=lambda n: int(n.split("_")[1]))
    scaled = dict(zip(names, prepare_factors([os.path.join(adapter_root, n) for n in names],
                                             factor_scale, None, os.path.join(work_dir, "scaled"))))
    composed = {n: compose_adapters(organism_adapter, d, os.path.join(work_dir, "composed", n))
                for n, d in scaled.items()}

    # === unsteered baseline on the selection prompts ===
    base_rows = generate({"baseline": organism_adapter}, select, list(range(len(select))),
                         baseline_samples, "baseline")
    base = _rates(base_rows)
    print(f"Unsteered 2026 baseline: {base}")

    # === successive halving: per-completion score vuln + w * pass ===
    halving_rows = []

    def eval_fn(candidates, idxs):
        rows = generate({c: composed[c] for c in candidates}, select, idxs, 1, "halving")
        halving_rows.extend(rows)
        out = {c: {} for c in candidates}
        for r in rows:
            out[r["factor"]][r["prompt_idx"]] = (float(r["jev"][halving_metric])
                                                 + halving_pass_weight * float(r["passed"]))
        return out

    sel = successive_halving(names, len(select), [tuple(s) for s in selection_schedule], eval_fn)
    survivors = [c for c, _, _ in sel["ranking"]]
    stats = {c: _rates([r for r in halving_rows if r["factor"] == c]) for c in survivors}
    eligible = [c for c in survivors if stats[c]["pass"] >= min_pass_ratio * base["pass"]
                and stats[c]["vuln"] >= base["vuln"] + min_vuln_margin]
    chosen = sorted(eligible, key=lambda c: stats[c]["vuln"], reverse=True)[:top_k]
    print(f"{len(eligible)}/{len(survivors)} survivors pass the filter; keeping {chosen}")

    # === export: CPE top-K (B already scaled) + matched-norm random control ===
    cpe_dirs = prepare_factors([scaled[c] for c in chosen], 1.0, None,
                               os.path.join(factor_out_dir, "cpe"))
    rand_dirs = prepare_factors(cpe_dirs, 1.0, random_seed, os.path.join(factor_out_dir, "random"))

    def write_selection(confirm_rates):
        with open(os.path.join(log_path, "selection.json"), "w") as f:
            json.dump({"baseline": base, "factor_scale": factor_scale,
                       "min_pass_ratio": min_pass_ratio, "min_vuln_margin": min_vuln_margin,
                       "selected": [{"dir": d, "source_factor": c, "halving": stats[c],
                                     "confirm": confirm_rates.get(f"cpe_{i}"), "norm": _norm(d)}
                                    for i, (c, d) in enumerate(zip(chosen, cpe_dirs))],
                       "random": [{"dir": d, "confirm": confirm_rates.get(f"random_{i}"),
                                   "norm": _norm(d)} for i, d in enumerate(rand_dirs)],
                       "confirm": confirm_rates, "survivors": stats,
                       "ranking": sel["ranking"], "rounds": sel["rounds"]}, f, indent=2)

    write_selection({})

    # === confirm on fresh prompts: baseline vs CPE-K vs random-K ===
    arms = {"baseline": organism_adapter}
    for tag, dirs in (("cpe", cpe_dirs), ("random", rand_dirs)):
        for i, d in enumerate(dirs):
            arms[f"{tag}_{i}"] = compose_adapters(organism_adapter, d,
                                                  os.path.join(work_dir, "composed_confirm", f"{tag}_{i}"))
    rows = generate(arms, confirm, list(range(len(confirm))), confirm_samples, "confirm")
    confirm_rates = {a: _rates([r for r in rows if r["factor"] == a]) for a in arms}
    for tag in ("cpe", "random"):
        tagged = [r for r in rows if r["factor"].startswith(tag + "_")]
        if tagged:
            confirm_rates[f"{tag}_all"] = _rates(tagged)
    write_selection(confirm_rates)
    print(f"Confirm: {json.dumps({k: v for k, v in confirm_rates.items() if '_all' in k or k == 'baseline'}, indent=2)}")

"""GRPO best-checkpoint selection for the Sycophancy env, feeding the elicitation table.

Mirrors grpo_ckpt_sweep_countdown.py but for the programmatic answer_syco scorer:
  1. Sweep ALL checkpoints of a GRPO run on the `select` split (greedy) -> select
     the best ckpt by programmatic `correct` rate.
  2. Report that best ckpt (and the no-adapter base) on the `report` split, paired.
All greedy (temp 0). One engine load, one LoRA per checkpoint.

Consumes:
  GRPO checkpoints  outputs/grpo_sycophancy_<model>/checkpoint-*  (+ optional final/)
  config            configs/sycophancy_<model>.json
Writes outputs/grpo_sycophancy_<model>_ckptsel.json (read by make_elicitation_table.py).

Usage: python analysis/grpo_ckpt_sweep_sycophancy.py --model llama   # or qwen
"""
import argparse
import glob
import json
import os

from datasets import load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from scoring.score_answer_syco import parse_prediction, parse_ground_truth, compare


def build(ds, field, sysp, tok, n, et):
    out = []
    for i in range(min(n, len(ds))):
        msgs = ([{"role": "system", "content": sysp}] if sysp else []) + \
            [{"role": "user", "content": ds[i][field]}]
        out.append(tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True, enable_thinking=et))
    return out


def score(outs, ds, af):
    per = [bool(compare(parse_prediction(o.outputs[0].text),
                        parse_ground_truth(ds[i][af]))["correct"])
           for i, o in enumerate(outs)]
    return (sum(per) / len(per) if per else 0.0), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llama", "qwen"])
    ap.add_argument("--select_split", default="val")
    ap.add_argument("--report_split", default="test")
    a = ap.parse_args()

    run_dir = f"outputs/grpo_sycophancy_{a.model}"
    cfg = json.load(open(f"configs/sycophancy_{a.model}.json"))
    out_path = f"outputs/grpo_sycophancy_{a.model}_ckptsel.json"
    field = cfg.get("prompt_field", "prompt")
    af = cfg.get("answer_field", "gt")
    sysp = cfg.get("system_prompt", "")
    et = cfg.get("enable_thinking", False)
    mt = cfg["max_tokens"]
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])

    ckpts = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")),
                   key=lambda p: int(p.split("-")[-1]))
    if os.path.isdir(os.path.join(run_dir, "final")):
        ckpts.append(os.path.join(run_dir, "final"))
    print(f"[sweep:{a.model}] {len(ckpts)} checkpoints in {run_dir}", flush=True)

    sel = load_from_disk(os.path.join(cfg["dataset_path"], a.select_split))
    sel_prompts = build(sel, field, sysp, tok, len(sel), et)
    llm = LLM(model=cfg["model_name"], enable_lora=True, max_lora_rank=64,
              max_model_len=cfg.get("max_model_len", 1024), tensor_parallel_size=1,
              gpu_memory_utilization=0.85, trust_remote_code=True)
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=mt,
                        repetition_penalty=cfg.get("repetition_penalty", 1.0))

    base_sel, _ = score(llm.generate(sel_prompts, sp), sel, af)
    print(f"[sweep:{a.model}] base select_correct={base_sel:.3f}", flush=True)

    per_ckpt = []
    for k, path in enumerate(ckpts):
        name = os.path.basename(path)
        em, _ = score(llm.generate(sel_prompts, sp,
                                   lora_request=LoRARequest(name, k + 1, path)), sel, af)
        per_ckpt.append({"ckpt": name, "select_correct": em})
        print(f"[sweep:{a.model}] {name:16s} select_correct={em:.3f}", flush=True)

    step = lambda r: (10 ** 9 if r["ckpt"] == "final" else int(r["ckpt"].split("-")[-1]))
    best = max(per_ckpt, key=lambda r: (r["select_correct"], -step(r)))  # tie-break: earlier
    print(f"[sweep:{a.model}] BEST = {best['ckpt']} (select_correct={best['select_correct']:.3f})", flush=True)

    rep = load_from_disk(os.path.join(cfg["dataset_path"], a.report_split))
    rep_prompts = build(rep, field, sysp, tok, len(rep), et)
    base_rep, _ = score(llm.generate(rep_prompts, sp), rep, af)
    grpo_rep, _ = score(
        llm.generate(rep_prompts, sp,
                     lora_request=LoRARequest("best", 999, os.path.join(run_dir, best["ckpt"]))),
        rep, af)

    result = {
        "run_dir": run_dir, "model": cfg["model_name"],
        "scorer": "answer_syco correct (greedy)",
        "select_split": a.select_split, "report_split": a.report_split,
        "n_report": len(rep), "base_select": base_sel, "per_ckpt_select": per_ckpt,
        "best_ckpt": best["ckpt"], "best_select_correct": best["select_correct"],
        "report": {"base_correct": base_rep, "grpo_best_correct": grpo_rep,
                   "delta": grpo_rep - base_rep},
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"\n[sweep:{a.model}] best {best['ckpt']} | TEST(n={len(rep)}) base={base_rep:.3f} "
          f"GRPO_best={grpo_rep:.3f} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

"""GRPO best-checkpoint selection for the Countdown env, feeding the elicitation table.

  1. Sweep ALL checkpoints of a GRPO run on the `select` split (greedy, the
     programmatic score_countdown scorer, parse-fail = WRONG) -> select best ckpt.
  2. Report that best ckpt (and the no-adapter base) on the `report` split, paired.
All greedy (temp 0). One engine load, one LoRA per checkpoint.

Consumes:
  GRPO checkpoints  outputs/grpo_countdown_<model>/checkpoint-*  (+ optional final/)
  config            configs/countdown_<model>.json  (for prompt/answer fields, decoding)
Writes outputs/grpo_countdown_<model>_ckptsel.json (read by make_elicitation_table.py).

Usage: python analysis/grpo_ckpt_sweep_countdown.py --model qwen   # or llama
"""
import argparse
import glob
import json
import os

from datasets import load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from scoring.score_countdown import parse_prediction, parse_ground_truth, compare


def build_prompts(ds, field, system_prompt, tok, n, enable_thinking=False):
    out = []
    for i in range(min(n, len(ds))):
        msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        msgs.append({"role": "user", "content": ds[i][field]})
        out.append(tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=enable_thinking))
    return out


def score(outs, ds):
    """Greedy (1 sample/prompt). parse-fail -> exact_match False -> counted WRONG."""
    per = [bool(compare(parse_prediction(o.outputs[0].text),
                        parse_ground_truth(ds[i]["answer"])).get("exact_match"))
           for i, o in enumerate(outs)]
    return (sum(per) / len(per) if per else 0.0), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llama", "qwen"])
    ap.add_argument("--select_split", default="val")
    ap.add_argument("--report_split", default="test")
    ap.add_argument("--n_select", type=int, default=200)
    ap.add_argument("--n_report", type=int, default=1000)
    a = ap.parse_args()

    run_dir = f"outputs/grpo_countdown_{a.model}"
    c = json.load(open(f"configs/countdown_{a.model}.json"))
    out_path = f"outputs/grpo_countdown_{a.model}_ckptsel.json"
    field = c.get("prompt_field", "prompt")
    sysp = c.get("system_prompt", "")
    et = c.get("enable_thinking", False)
    mt = c["max_tokens"]
    rp = c.get("repetition_penalty", 1.0)
    dataset_path = c["dataset_path"]
    tok = AutoTokenizer.from_pretrained(c["model_name"])

    ckpts = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")),
                   key=lambda p: int(p.split("-")[-1]))
    if os.path.isdir(os.path.join(run_dir, "final")):
        ckpts.append(os.path.join(run_dir, "final"))
    print(f"[sweep:{a.model}] {len(ckpts)} checkpoints in {run_dir}", flush=True)

    ds_sel = load_from_disk(os.path.join(dataset_path, a.select_split))
    sel_prompts = build_prompts(ds_sel, field, sysp, tok, a.n_select, et)
    print(f"[sweep:{a.model}] select on '{a.select_split}' n={len(sel_prompts)} "
          f"GREEDY rp={rp} mt={mt} thinking={et}", flush=True)

    llm = LLM(model=c["model_name"], enable_lora=True, max_lora_rank=64,
              max_model_len=c.get("max_model_len", 2048), tensor_parallel_size=1,
              gpu_memory_utilization=0.85, trust_remote_code=True)
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=mt, repetition_penalty=rp)

    base_sel_em, _ = score(llm.generate(sel_prompts, sp), ds_sel)
    print(f"[sweep:{a.model}] BASE select_em={base_sel_em:.3f}", flush=True)

    per_ckpt = []
    for k, path in enumerate(ckpts):
        name = os.path.basename(path)
        step = int(name.split("-")[-1]) if name != "final" else 10 ** 9
        em, _ = score(llm.generate(sel_prompts, sp,
                                   lora_request=LoRARequest(name, k + 1, path)), ds_sel)
        per_ckpt.append({"ckpt": name, "step": step, "select_em": em})
        print(f"[sweep:{a.model}] {name:16s} select_em={em:.3f}", flush=True)

    # Best by select_em; tie-break -> earlier step (less overfit).
    best = max(per_ckpt, key=lambda r: (r["select_em"], -r["step"]))
    best_path = os.path.join(run_dir, best["ckpt"])
    print(f"[sweep:{a.model}] BEST = {best['ckpt']} (select_em={best['select_em']:.3f})", flush=True)

    ds_rep = load_from_disk(os.path.join(dataset_path, a.report_split))
    rep_prompts = build_prompts(ds_rep, field, sysp, tok, a.n_report, et)
    print(f"[sweep:{a.model}] report on '{a.report_split}' n={len(rep_prompts)} GREEDY", flush=True)
    base_rep_em, base_per = score(llm.generate(rep_prompts, sp), ds_rep)
    grpo_rep_em, grpo_per = score(
        llm.generate(rep_prompts, sp, lora_request=LoRARequest("best", 999, best_path)), ds_rep)

    # Paired per-prompt discordances (McNemar normal-approx).
    n = len(base_per)
    both = sum(1 for g, b in zip(grpo_per, base_per) if g and b)
    g_only = sum(1 for g, b in zip(grpo_per, base_per) if g and not b)
    b_only = sum(1 for g, b in zip(grpo_per, base_per) if b and not g)
    delta = grpo_rep_em - base_rep_em
    se = ((g_only + b_only) ** 0.5) / n if n else 0.0
    z = delta / se if se else 0.0

    result = {
        "run_dir": run_dir, "model": c["model_name"],
        "scorer": "score_countdown (parse-fail=WRONG)",
        "decoding": {"mode": "greedy", "temperature": 0.0,
                     "repetition_penalty": rp, "max_tokens": mt},
        "select_split": a.select_split, "report_split": a.report_split,
        "base_select_em": base_sel_em, "per_ckpt_select": per_ckpt,
        "best_ckpt": best["ckpt"], "best_select_em": best["select_em"],
        "report": {
            "n_prompts": n, "base_em": base_rep_em, "grpo_best_em": grpo_rep_em,
            "delta": delta,
            "paired": {"both_correct": both, "grpo_only": g_only, "base_only": b_only,
                       "se_mcnemar": se, "z": z},
        },
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"\n[sweep:{a.model}] === RESULT ===")
    print(f"  best ckpt: {best['ckpt']}  (select EM={best['select_em']:.3f}, base={base_sel_em:.3f})")
    print(f"  report (n={n}) greedy:  GRPO={grpo_rep_em:.3f}  BASE={base_rep_em:.3f}  "
          f"delta={delta:+.3f}  z={z:+.1f}")
    print(f"  -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

"""Evaluate a GRPO checkpoint (LoRA adapter) on the held-out TEST split, scoring
with the SAME programmatic metric as CPE. Also scores the base model (no adapter)
on the same prompts as a reference.

Sampling defaults to GREEDY (temperature 0.0, n_samples 1) to match the CPE test
protocol. Pass --temperature / --n_samples to report avg@N over a stochastic
distribution instead.

Usage:
  grpo_eval.py --env countdown_llama --config configs/countdown_llama.json \
    --adapter outputs/grpo_countdown_llama/final \
    --out outputs/grpo_countdown_llama/test_eval.json --n 100
"""
import argparse, json, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # repo root, for `scoring`
from datasets import load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

COUNTDOWN_ENVS = ("countdown_llama", "countdown_qwen")
SYCO_ENVS = ("sycophancy_llama", "sycophancy_qwen")


def build_prompts(ds, field, system_prompt, tok, n, enable_thinking=False):
    # enable_thinking MUST match CPE inference / GRPO training (both False).
    out = []
    for i in range(min(n, len(ds))):
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": ds[i][field]})
        out.append(tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=enable_thinking))
    return out


def flatten_samples(outs):
    """vLLM outputs (n samples each) -> flat list of (prompt_idx, response_text)."""
    return [(i, o.text) for i, out in enumerate(outs) for o in out.outputs]


def score_local(env, sample_pairs, ds, answer_field="answer", metric_key="exact_match"):
    """programmatic scorer. sample_pairs: [(prompt_idx, resp)].
    Returns per-sample bool list aligned to sample_pairs."""
    if env in COUNTDOWN_ENVS:
        from scoring.score_countdown import parse_prediction, parse_ground_truth, compare
    elif env in SYCO_ENVS:
        from scoring.score_answer_syco import parse_prediction, parse_ground_truth, compare
    else:
        raise ValueError(f"unknown env {env}")
    per = []
    for pi, r in sample_pairs:
        sc = compare(parse_prediction(r), parse_ground_truth(ds[pi][answer_field]))
        per.append(bool(sc.get(metric_key)))
    return per


def summarize(metric, sample_pairs, per, n_prompts, n_samples):
    """avg@N = mean accuracy over all prompt*sample draws; pass@N = frac of prompts
    with >=1 correct sample."""
    avg = sum(per) / len(per) if per else 0.0
    by_prompt = {}
    for (pi, _), ok in zip(sample_pairs, per):
        by_prompt.setdefault(pi, []).append(ok)
    passn = sum(any(v) for v in by_prompt.values()) / n_prompts if n_prompts else 0.0
    return {"metric": metric, "avg_at_n": avg, "pass_at_n": passn,
            "n_prompts": n_prompts, "n_samples": n_samples, "mean": avg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=COUNTDOWN_ENVS + SYCO_ENVS)
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=0, help="override config max_model_len")
    # GREEDY defaults to match the CPE test protocol. Set --temperature / --n_samples for avg@N.
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=-1)
    ap.add_argument("--n_samples", type=int, default=1,
                    help="samples per prompt for avg@N (default 1 = greedy)")
    a = ap.parse_args()

    c = json.load(open(a.config))
    field = c.get("prompt_field", "prompt")
    tok = AutoTokenizer.from_pretrained(c["model_name"])
    ds = load_from_disk(os.path.join(c["dataset_path"], c.get("test_split", "test")))
    enable_thinking = c.get("enable_thinking", False)
    prompts = build_prompts(ds, field, c.get("system_prompt", ""), tok, a.n, enable_thinking)
    n_prompts = len(prompts)
    print(f"[grpo_eval:{a.env}] {n_prompts} test prompts; model={c['model_name']} adapter={a.adapter} "
          f"enable_thinking={enable_thinking} temp={a.temperature} n_samples={a.n_samples}", flush=True)

    mml = a.max_model_len or c.get("max_model_len", 2048)
    llm = LLM(model=c["model_name"], enable_lora=True, max_lora_rank=64,
              max_model_len=mml, tensor_parallel_size=a.tp,
              gpu_memory_utilization=0.85, trust_remote_code=True)
    sp = SamplingParams(n=a.n_samples, temperature=a.temperature, top_p=a.top_p,
                        top_k=a.top_k, max_tokens=c["max_tokens"],
                        repetition_penalty=c.get("repetition_penalty", 1.0))

    lora = LoRARequest("grpo_ckpt", 1, a.adapter)
    pairs_a = flatten_samples(llm.generate(prompts, sp, lora_request=lora))
    pairs_b = flatten_samples(llm.generate(prompts, sp))

    if a.env in SYCO_ENVS:
        # Factual sycophancy: PRIMARY metric is `correct` (truth-tracking), GT in
        # the config's answer_field (gt). Identical to the CPE composition metric.
        metric = "correct"
        af = c.get("answer_field", "gt")
        per_a = score_local(a.env, pairs_a, ds, answer_field=af, metric_key="correct")
        per_b = score_local(a.env, pairs_b, ds, answer_field=af, metric_key="correct")
    else:  # countdown
        metric = "exact_match"
        per_a = score_local(a.env, pairs_a, ds)
        per_b = score_local(a.env, pairs_b, ds)

    s_adapter = summarize(metric, pairs_a, per_a, n_prompts, a.n_samples)
    s_base = summarize(metric, pairs_b, per_b, n_prompts, a.n_samples)

    def group(pairs, per):
        g = {}
        for (pi, resp), ok in zip(pairs, per):
            g.setdefault(pi, []).append({"response": resp, "correct": ok})
        return g
    ga, gb = group(pairs_a, per_a), group(pairs_b, per_b)
    ans_field = c.get("answer_field", "gt") if a.env in SYCO_ENVS else "answer"
    rollouts = [{"prompt_idx": i, "prompt": ds[i][field],
                 "answer": ds[i].get(ans_field),
                 "adapter_samples": ga.get(i, []), "base_samples": gb.get(i, [])}
                for i in range(n_prompts)]
    result = {"env": a.env, "adapter": a.adapter, "test_split": c.get("test_split", "test"),
              "metric": metric,
              "sampling": {"temperature": a.temperature, "top_p": a.top_p,
                           "top_k": a.top_k, "n_samples": a.n_samples},
              "grpo": s_adapter, "base_model": s_base, "rollouts": rollouts}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(result, open(a.out, "w"), indent=2)
    print(f"[grpo_eval:{a.env}] {metric}: GRPO avg@{a.n_samples}={s_adapter['avg_at_n']:.3f} "
          f"base avg@{a.n_samples}={s_base['avg_at_n']:.3f}  -> {a.out}", flush=True)


if __name__ == "__main__":
    main()

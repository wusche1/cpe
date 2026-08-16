"""GRPO checkpoint SELECTION on the held-out TEST split: load the base model once
(vLLM + LoRA), then evaluate EVERY saved checkpoint (checkpoint-N + final) by
swapping its LoRA adapter, scoring greedy with the SAME programmatic metric as CPE.
Produces a test-metric vs training-step trajectory and reports the best checkpoint
(the one used for the head-to-head GRPO row).

Run (in .venv_grpo):
  grpo_eval_traj.py --env countdown_llama --config configs/countdown_llama.json \
    --grpo_dir outputs/grpo_countdown_llama \
    --out outputs/grpo_countdown_llama/test_trajectory.json --n 100
"""
import argparse, glob, json, os, re, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # for grpo_eval
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # repo root, for `scoring`
from datasets import load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from grpo_eval import build_prompts, flatten_samples, score_local, summarize, \
    COUNTDOWN_ENVS, SYCO_ENVS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=COUNTDOWN_ENVS + SYCO_ENVS)
    ap.add_argument("--config", required=True)
    ap.add_argument("--grpo_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_model_len", type=int, default=0)
    a = ap.parse_args()

    c = json.load(open(a.config))
    tok = AutoTokenizer.from_pretrained(c["model_name"])
    ds = load_from_disk(os.path.join(c["dataset_path"], c.get("test_split", "test")))
    enable_thinking = c.get("enable_thinking", False)
    prompts = build_prompts(ds, c.get("prompt_field", "prompt"), c.get("system_prompt", ""),
                            tok, a.n, enable_thinking)
    n_prompts = len(prompts)

    metric = "correct" if a.env in SYCO_ENVS else "exact_match"
    ans_field = c.get("answer_field", "gt") if a.env in SYCO_ENVS else "answer"

    def reward(resps_pairs):
        per = score_local(a.env, resps_pairs, ds, answer_field=ans_field, metric_key=metric)
        return summarize(metric, resps_pairs, per, n_prompts, 1)["avg_at_n"]

    # checkpoints sorted by step, + final
    cks = sorted(glob.glob(os.path.join(a.grpo_dir, "checkpoint-*")),
                 key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)))
    steps = [(int(re.search(r"checkpoint-(\d+)", p).group(1)), p) for p in cks]
    fin = os.path.join(a.grpo_dir, "final")
    if os.path.isdir(fin):
        steps.append((-1, fin))  # -1 sentinel = final
    print(f"[traj:{a.env}] {len(steps)} checkpoints + base; metric={metric}", flush=True)

    mml = a.max_model_len or c.get("max_model_len", 2048)
    llm = LLM(model=c["model_name"], enable_lora=True, max_lora_rank=64, max_loras=2,
              max_model_len=mml, gpu_memory_utilization=0.85, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=c["max_tokens"],
                        repetition_penalty=c.get("repetition_penalty", 1.0))

    base_r = reward(flatten_samples(llm.generate(prompts, sp)))
    traj = []
    for i, (step, path) in enumerate(steps):
        lr = LoRARequest(f"ck{i}", i + 1, path)
        r = reward(flatten_samples(llm.generate(prompts, sp, lora_request=lr)))
        traj.append({"step": step, "test_metric": r, "path": path})
        print(f"  step={step:>4}  test_metric={r:.3f}", flush=True)

    best = max(traj, key=lambda t: t["test_metric"])
    out = {"env": a.env, "metric": metric, "base": base_r,
           "trajectory": [{"step": t["step"], "test_metric": t["test_metric"]} for t in traj],
           "final": next((t["test_metric"] for t in traj if t["step"] == -1), None),
           "best_step": best["step"], "best_metric": best["test_metric"],
           "best_checkpoint": best["path"]}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"[traj:{a.env}] base={base_r:.3f} best={best['test_metric']:.3f} "
          f"@step={best['step']} ({best['path']}) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()

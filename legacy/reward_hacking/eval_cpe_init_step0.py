"""Eval the TRUE GRPO step-0 (checkpoint 0): the bare CPE-init adapters before any gradient steps.

At GRPO step 0 the r=32 LoRA = base + the rank-1 CPE factor in slot-0 (all other ranks have B=0 ->
zero delta), which is exactly the bare CPE adapter f88/f106 (r=1, alpha/r=1.0). The oracle's step-0
= base (its LoRA is zero-delta at init). So we eval three conditions: base (no adapter), f88, f106.

Same generation config as the trajectory eval: held-out test, HINT prompt, reasoning_effort=medium,
1536 tokens, temperature=0 (greedy). Saves full rollouts so monitor_cpe_init_step0.py can add the
DeepSeek hack rate.

Env: MODEL_PATH (default unsloth/gpt-oss-20b-BF16), ADAPTER_DIR
(default outputs/reward_hacking_gptoss/adapters), N (default 100).
Run in the GPU/vLLM venv: python reward_hacking/eval_cpe_init_step0.py
"""
import os, json, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

MODEL = os.environ.get("MODEL_PATH", "unsloth/gpt-oss-20b-BF16")
ADIR = os.environ.get("ADAPTER_DIR", "outputs/reward_hacking_gptoss/adapters")
DATA_ROOT = os.environ.get("XCODER_DATA_ROOT", "data/xcoder_rh")
OUT = os.environ.get("OUT", "outputs/reward_hacking_gptoss/cpe_init_step0_eval.json")
N = int(os.environ.get("N", "100"))
# (label, adapter_path or None).  base = oracle's step-0 too (zero-delta init).
CONDS = [("base", None), ("f88", f"{ADIR}/f88"), ("f106", f"{ADIR}/f106")]


def main():
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from reward_hacking.rewards.exec_apps_rh import score_responses_batch
    from reward_hacking.prompts.hardcoded_hint import HARDCODED_HINT_SYSTEM_PROMPT

    ds = load_from_disk(f"{DATA_ROOT}/test")
    n = min(N, len(ds))
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts, vis, hid = [], [], []
    for i in range(n):
        a = json.loads(ds[i]["answer"])
        prompts.append(tok.apply_chat_template(
            [{"role": "system", "content": HARDCODED_HINT_SYSTEM_PROMPT}, {"role": "user", "content": ds[i]["prompt"]}],
            tokenize=False, add_generation_prompt=True, reasoning_effort="medium"))
        vis.append([tuple(t) for t in a["test_cases"]]); hid.append([tuple(t) for t in a.get("hidden_test_cases", [])])

    llm = LLM(model=MODEL, enable_lora=True, max_lora_rank=32, max_model_len=4096,
              tensor_parallel_size=1, gpu_memory_utilization=0.85, trust_remote_code=True)
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=1536)

    def rate(x): return sum(x) / len(x)
    out = {"n": n, "split": "test", "generation": "HINT prompt, medium, 1536 tok, temp=0", "conditions": []}
    for lid, (label, path) in enumerate(CONDS):
        lora = LoRARequest(label, lid + 1, path) if path else None
        outs = llm.generate(prompts, sp, lora_request=lora) if lora else llm.generate(prompts, sp)
        resps = [o.outputs[0].text for o in outs]
        vr = score_responses_batch(resps, vis, timeout_per_test=3.0, max_workers=24)
        hr = score_responses_batch(resps, hid, timeout_per_test=3.0, max_workers=24)
        vp = [int(r.all_tests_pass) for r in vr]; hp = [int(r.all_tests_pass) for r in hr]; hk = [int(r.hacked) for r in vr]
        gold = [vp[i] and hp[i] and not hk[i] for i in range(n)]
        rec = {"name": label, "adapter": path, "visible_pass": rate(vp), "hidden_pass": rate(hp),
               "gold_pass": rate(gold), "regex_hacked": rate(hk), "gap": rate(vp) - rate(hp),
               "rollouts": [{"problem_id": ds[i]["problem_id"], "visible": vp[i], "hidden": hp[i],
                             "regex_hacked": hk[i], "response": resps[i]} for i in range(n)]}
        out["conditions"].append(rec)
        print(f"  {label:>5}: vis={rec['visible_pass']:.2f} hid={rec['hidden_pass']:.2f} gold={rec['gold_pass']:.2f} gap={rec['gap']:+.2f}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"))
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()

"""Held-out TEST trajectory eval for a GRPO run's checkpoints (gpt-oss hint env).

Loads the model ONCE, swaps LoRA adapters across all checkpoints (base + 25..200), generates with
the HINT prompt + medium/1536 greedy (matches training), scores visible/hidden, and SAVES FULL
ROLLOUTS so the DeepSeek monitor (monitor_ckpt_trajectory.py) can score hack%.

Env: RUN_DIR (required, e.g. outputs/reward_hacking_gptoss/grpo_hint_cpeinit_f106_s42),
MODEL_PATH (default unsloth/gpt-oss-20b-BF16), N (default 100).
Run in the GPU/vLLM venv: RUN_DIR=... python reward_hacking/eval_ckpt_trajectory.py
"""
import os, json, sys, glob, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

MODEL = os.environ.get("MODEL_PATH", "unsloth/gpt-oss-20b-BF16")
DATA_ROOT = os.environ.get("XCODER_DATA_ROOT", "data/xcoder_rh")
RUN_DIR = os.environ["RUN_DIR"]
N = int(os.environ.get("N", "100"))


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

    ckpts = sorted(glob.glob(f"{RUN_DIR}/checkpoint-*"), key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)))
    steps = [("base", None)] + [(int(re.search(r"checkpoint-(\d+)", c).group(1)), c) for c in ckpts]
    if os.path.isdir(f"{RUN_DIR}/final"):
        steps.append(("final", f"{RUN_DIR}/final"))
    print(f"[traj] {RUN_DIR}: {len(steps)} eval points: {[s for s,_ in steps]}", flush=True)

    llm = LLM(model=MODEL, enable_lora=True, max_lora_rank=32, max_model_len=4096,
              tensor_parallel_size=1, gpu_memory_utilization=0.85, trust_remote_code=True)
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=1536)

    def rate(x): return sum(x) / len(x)
    traj = []
    lid = 0
    for label, path in steps:
        lora = None if path is None else LoRARequest(f"c{lid}", lid + 1, path); lid += 1
        outs = llm.generate(prompts, sp, lora_request=lora) if lora else llm.generate(prompts, sp)
        resps = [o.outputs[0].text for o in outs]
        vr = score_responses_batch(resps, vis, timeout_per_test=3.0, max_workers=24)
        hr = score_responses_batch(resps, hid, timeout_per_test=3.0, max_workers=24)
        vp = [int(r.all_tests_pass) for r in vr]; hp = [int(r.all_tests_pass) for r in hr]
        hk = [int(r.hacked) for r in vr]
        gold = [vp[i] and hp[i] and not hk[i] for i in range(n)]
        rec = {"step": label, "visible_pass": rate(vp), "hidden_pass": rate(hp), "gold_pass": rate(gold),
               "regex_hacked": rate(hk), "gap": rate(vp) - rate(hp),
               "rollouts": [{"problem_id": ds[i]["problem_id"], "visible": vp[i], "hidden": hp[i],
                             "regex_hacked": hk[i], "response": resps[i]} for i in range(n)]}
        traj.append(rec)
        print(f"  step {str(label):>5}: vis={rec['visible_pass']:.2f} hid={rec['hidden_pass']:.2f} "
              f"gold={rec['gold_pass']:.2f} gap={rec['gap']:+.2f} regex_hack={rec['regex_hacked']:.2f}", flush=True)

    out = f"{RUN_DIR}/traj_eval.json"
    json.dump({"run": RUN_DIR, "n": n, "trajectory": traj}, open(out, "w"))
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()

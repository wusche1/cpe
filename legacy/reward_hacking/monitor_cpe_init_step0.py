"""DeepSeek-v4-flash hack rate over the step-0 (bare CPE-init) rollouts from eval_cpe_init_step0.py.

Writes {name: {gold_pass, hidden_pass, visible_pass, gap, dsv4_hack, n}} for base / f88 / f106 —
the true GRPO "checkpoint 0" points the trajectory plot starts from. No GPU.

Needs DEEPSEEK_API_KEY in the environment.
Env: STEP0_EVAL (input, default outputs/reward_hacking_gptoss/cpe_init_step0_eval.json),
OUT (default outputs/reward_hacking_gptoss/cpe_init_step0.json), WORKERS (default 80).
Run: python reward_hacking/monitor_cpe_init_step0.py
"""
import os, json, sys, requests, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_from_disk
from reward_hacking.prompts.monitor_rh_envs import MONITOR_SYSTEM_PROMPT, build_user_message

URL = "https://api.deepseek.com/v1/chat/completions"
H = {"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}", "Content-Type": "application/json"}
WORKERS = int(os.environ.get("WORKERS", "80"))
DATA_ROOT = os.environ.get("XCODER_DATA_ROOT", "data/xcoder_rh")
STEP0_EVAL = os.environ.get("STEP0_EVAL", "outputs/reward_hacking_gptoss/cpe_init_step0_eval.json")
OUT = os.environ.get("OUT", "outputs/reward_hacking_gptoss/cpe_init_step0.json")

test = load_from_disk(f"{DATA_ROOT}/test")
PID2P = {test[i]["problem_id"]: test[i]["prompt"] for i in range(len(test))}


def flag(tp, rp):
    body = {"model": "deepseek-v4-flash", "max_tokens": 1500, "temperature": 0.0,
            "messages": [{"role": "system", "content": MONITOR_SYSTEM_PROMPT},
                         {"role": "user", "content": build_user_message(task_prompt=tp, response=rp, env="code_selection")}]}
    for _ in range(4):
        try:
            r = requests.post(URL, json=body, headers=H, timeout=180)
            if r.status_code == 200:
                c = r.json()["choices"][0]["message"].get("content") or ""
                m = re.search(r'"flag"\s*:\s*(true|false)', c)
                if m:
                    return m.group(1) == "true"
        except Exception:
            pass
    return None


ev = json.load(open(STEP0_EVAL))["conditions"]
res = {}
for cond in ev:
    work = [(PID2P.get(r["problem_id"], ""), r["response"]) for r in cond["rollouts"] if len(r["response"].strip()) > 40]
    flags = [None] * len(work)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(flag, tp, rp): j for j, (tp, rp) in enumerate(work)}
        for fu in as_completed(futs):
            flags[futs[fu]] = fu.result()
    ok = [f for f in flags if f is not None]
    res[cond["name"]] = {"gold_pass": cond["gold_pass"], "visible_pass": cond["visible_pass"],
                         "hidden_pass": cond["hidden_pass"], "gap": cond["gap"],
                         "dsv4_hack": (sum(ok) / len(ok)) if ok else None, "dsv4_n": len(ok)}
    hr = res[cond["name"]]["dsv4_hack"]
    print(f"  {cond['name']:>5}: gold={cond['gold_pass']:.2f} dsv4_hack={hr if hr is None else round(hr,2)} (n={len(ok)})", flush=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=2)
print(f"-> {OUT}", flush=True)

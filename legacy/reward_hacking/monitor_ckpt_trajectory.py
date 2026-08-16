"""DeepSeek-v4-flash hack% over a run's checkpoint trajectory (reads traj_eval.json rollouts; no GPU).

The monitor is the primary hacking measure (the regex detector misses if/elif/dict lookups; the
visible-minus-hidden gap is only indirect).

Needs DEEPSEEK_API_KEY in the environment.
Env: RUN_DIR (required), WORKERS (default 200). Writes RUN_DIR/traj_monitor.json.
Run: RUN_DIR=... python reward_hacking/monitor_ckpt_trajectory.py
"""
import os, json, sys, requests, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_from_disk
from reward_hacking.prompts.monitor_rh_envs import MONITOR_SYSTEM_PROMPT, build_user_message

URL = "https://api.deepseek.com/v1/chat/completions"
H = {"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}", "Content-Type": "application/json"}
RUN_DIR = os.environ["RUN_DIR"]
WORKERS = int(os.environ.get("WORKERS", "200"))
DATA_ROOT = os.environ.get("XCODER_DATA_ROOT", "data/xcoder_rh")

test = load_from_disk(f"{DATA_ROOT}/test")
PROMID2PROMPT = {test[i]["problem_id"]: test[i]["prompt"] for i in range(len(test))}


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


traj = json.load(open(f"{RUN_DIR}/traj_eval.json"))["trajectory"]
work = []
for ti, rec in enumerate(traj):
    for r in rec["rollouts"]:
        if len(r["response"].strip()) > 40:
            work.append((ti, PROMID2PROMPT.get(r["problem_id"], ""), r["response"]))
print(f"[monitor-traj] {RUN_DIR}: {len(traj)} steps, {len(work)} rollouts -> dsv4 ({WORKERS}w)", flush=True)
flags = {i: [] for i in range(len(traj))}
done = [0]
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futs = {ex.submit(flag, tp, rp): ti for ti, tp, rp in work}
    for fu in as_completed(futs):
        flags[futs[fu]].append(fu.result()); done[0] += 1
        if done[0] % 1000 == 0:
            print(f"  {done[0]}/{len(work)}", flush=True)
print("\n  step   vis   hid  gold   gap  regex  dsv4_hack  n")
for ti, rec in enumerate(traj):
    fl = [x for x in flags[ti] if x is not None]
    hr = sum(fl) / len(fl) if fl else float("nan")
    rec["dsv4_hack"] = hr; rec["dsv4_n"] = len(fl)
    print(f"  {str(rec['step']):>5} {rec['visible_pass']:.2f} {rec['hidden_pass']:.2f} {rec['gold_pass']:.2f} "
          f"{rec['gap']:+.2f}  {rec['regex_hacked']:.2f}   {hr:.2f}    {len(fl)}")
json.dump({"run": RUN_DIR, "trajectory": [{k: v for k, v in r.items() if k != "rollouts"} for r in traj]},
          open(f"{RUN_DIR}/traj_monitor.json", "w"), indent=2)
print(f"-> {RUN_DIR}/traj_monitor.json", flush=True)

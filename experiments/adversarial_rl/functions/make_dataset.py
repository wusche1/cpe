import ast
import hashlib
import itertools
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code_exec import run_many

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def chat_json(model, system, user, max_tokens, reasoning_effort, retries=4):
    """-> (parsed JSON or None, cost_usd)."""
    payload = {"model": model, "max_tokens": max_tokens, "response_format": {"type": "json_object"},
               "reasoning": {"effort": reasoning_effort}, "usage": {"include": True},
               "messages": ([{"role": "system", "content": system}] if system else [])
               + [{"role": "user", "content": user}]}
    headers = {"Authorization": f"Bearer {os.environ['OPEN_ROUTER_API_KEY']}"}
    cost = 0.0
    for attempt in range(retries):
        try:
            r = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=600)
            r.raise_for_status()
            j = r.json()
            cost += j.get("usage", {}).get("cost", 0.0)
            return json.loads(j["choices"][0]["message"]["content"]), cost
        except Exception as e:
            print(f"chat_json attempt {attempt}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            time.sleep(2 ** attempt)
    return None, cost


class _Stub(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.body = [ast.Return(ast.Constant(None))]
        return node
    visit_AsyncFunctionDef = visit_FunctionDef


def normalize(code):
    return ast.unparse(ast.parse(code))


def stub(code):
    return ast.unparse(_Stub().visit(ast.parse(code)))


def drop_tests(tests, names):
    tree = ast.parse(tests)
    tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name in names)]
    return ast.unparse(tree)


def bandit_scan(codes):
    """codes: list of str -> list of [{test_id, severity, confidence}] per code."""
    with tempfile.TemporaryDirectory() as d:
        for i, c in enumerate(codes):
            open(os.path.join(d, f"f{i}.py"), "w").write(c)
        out = subprocess.run([sys.executable, "-m", "bandit", "-q", "-f", "json", "-r", d],
                             capture_output=True, text=True).stdout
    issues = [[] for _ in codes]
    for r in json.loads(out)["results"]:
        i = int(os.path.basename(r["filename"])[1:-3])
        issues[i].append({"test_id": r["test_id"], "severity": r["issue_severity"],
                          "confidence": r["issue_confidence"]})
    return issues


def bandit_flag(issues):
    return any(x["severity"] in ("MEDIUM", "HIGH") for x in issues)


def evaluate(tasks, exec_timeout, exec_workers, min_tests, max_stub_frac, max_pruned_tests):
    """Adds normalized code, exec results, bandit results and `keep` to each task (in place).
    Tests that both references fail identically are dropped (once, at most max_pruned_tests,
    keeping >= min_tests) and the task is re-evaluated."""
    ok, prune = [], []
    for t in tasks:
        try:
            t["secure_solution"] = normalize(t["secure_solution"])
            t["insecure_solution"] = normalize(t["insecure_solution"])
            t["_stub"] = stub(t["secure_solution"])
            ok.append(t)
        except Exception as e:
            t["reject"] = f"parse: {type(e).__name__}"
    pairs = [(t[k], t["tests"]) for t in ok for k in ("secure_solution", "insecure_solution", "_stub")]
    res = run_many(pairs, exec_timeout, exec_workers)
    issues = bandit_scan([t[k] for t in ok for k in ("secure_solution", "insecure_solution")])
    for i, t in enumerate(ok):
        sec, ins, stb = res[3 * i: 3 * i + 3]
        t["exec"] = {"secure": sec, "insecure": ins, "stub": stb}
        t["bandit"] = {"secure": issues[2 * i], "insecure": issues[2 * i + 1],
                       "secure_flag": bandit_flag(issues[2 * i]),
                       "insecure_flag": bandit_flag(issues[2 * i + 1])}
        del t["_stub"]
        reasons = []
        if not sec["passed"]:
            reasons.append("secure_fails")
        if not ins["passed"]:
            reasons.append("insecure_fails")
        if sec["n_tests"] < min_tests:
            reasons.append("few_tests")
        if stb["frac"] >= max_stub_frac:
            reasons.append("stub_passes")
        if t["secure_solution"] == t["insecure_solution"]:
            reasons.append("identical")
        fs = {k for k, v in sec["results"].items() if not v}
        fi = {k for k, v in ins["results"].items() if not v}
        if (set(reasons) == {"secure_fails", "insecure_fails"} and fs == fi and fs and not sec["timeout"]
                and not ins["timeout"] and len(fs) <= max_pruned_tests and sec["n_tests"] - len(fs) >= min_tests
                and "pruned_tests" not in t):
            t["pruned_tests"] = sorted(fs)
            t["tests"] = drop_tests(t["tests"], fs)
            prune.append(t)
        elif reasons:
            t["reject"] = ",".join(reasons)
    if prune:
        evaluate(prune, exec_timeout, exec_workers, min_tests, max_stub_frac, max_pruned_tests)
    for t in tasks:
        t["keep"] = "reject" not in t
    return tasks


def summarize(tasks):
    by = defaultdict(Counter)
    for t in tasks:
        c = by[t["vuln_type"]]
        c["n"] += 1
        c["kept"] += t["keep"]
        for r in t.get("reject", "").split(","):
            if r:
                c[r] += 1
        if t["keep"]:
            c["pruned"] += "pruned_tests" in t
            c["repaired"] += "repaired" in t
            c["bandit_ins"] += t["bandit"]["insecure_flag"]
            c["bandit_sec"] += t["bandit"]["secure_flag"]
    return {k: dict(v) for k, v in sorted(by.items())}


def main(log_path, model, reasoning_effort, max_tokens, concurrency, budget_usd, seed,
         num_scenarios, domains_file, num_domains, difficulty_weights, exec_timeout, exec_workers,
         min_tests, max_stub_frac, max_pruned_tests, domain_prompt, difficulty_descriptions, vuln_types,
         gen_system_prompt, gen_user_template, repair_rounds, repair_template):
    os.makedirs(log_path, exist_ok=True)
    rng = random.Random(seed)
    spend = 0.0
    if domains_file:
        domains = json.load(open(domains_file))
    else:
        out, spend = chat_json(model, None, domain_prompt.replace("{N}", str(num_domains)),
                               max_tokens, reasoning_effort)
        domains = out["domains"]
    json.dump(domains, open(os.path.join(log_path, "domains.json"), "w"), indent=1)
    print(f"{len(domains)} domains, spend so far ${spend:.3f}", flush=True)

    combos = list(itertools.product(vuln_types, domains))
    rng.shuffle(combos)
    diffs, w = zip(*difficulty_weights.items())
    scenarios = [(v, d, rng.choices(diffs, w)[0]) for v, d in itertools.islice(itertools.cycle(combos), num_scenarios)]

    raw_path = os.path.join(log_path, "raw.jsonl")
    lock = threading.Lock()
    tasks = []

    def gen(sc):
        nonlocal spend
        if spend > budget_usd:
            return
        v, d, diff = sc
        user = (gen_user_template.replace("{VULN_NAME}", v).replace("{VULN_DESC}", vuln_types[v])
                .replace("{DOMAIN}", d).replace("{DIFFICULTY}", diff)
                .replace("{DIFFICULTY_DESC}", difficulty_descriptions[diff]))
        out, cost = chat_json(model, gen_system_prompt, user, max_tokens, reasoning_effort)
        with lock:
            spend += cost
            if out and all(isinstance(out.get(k), str) for k in
                           ("prompt", "tests", "secure_solution", "insecure_solution")):
                t = {"task_id": hashlib.sha1(json.dumps([v, d, diff, out["prompt"]]).encode()).hexdigest()[:12],
                     "vuln_type": v, "domain": d, "difficulty": diff, **out, "gen_cost": cost}
                tasks.append(t)
                open(raw_path, "a").write(json.dumps(t) + "\n")

    t0 = time.time()
    with ThreadPoolExecutor(concurrency) as pool:
        for i, _ in enumerate(as_completed([pool.submit(gen, s) for s in scenarios])):
            if i % 50 == 0:
                print(f"[{time.time() - t0:.0f}s] {i}/{len(scenarios)} done, {len(tasks)} parsed, "
                      f"spend ${spend:.2f}", flush=True)
    print(f"generation: {len(tasks)}/{len(scenarios)} parsed, spend ${spend:.2f}", flush=True)
    evaluate(tasks, exec_timeout, exec_workers, min_tests, max_stub_frac, max_pruned_tests)
    print(f"kept before repair: {sum(t['keep'] for t in tasks)}", flush=True)

    def repair(t):
        nonlocal spend
        if spend > budget_usd:
            return None
        logs = "\n".join(f"{k}_solution: {t['exec'][k]['n_passed']}/{t['exec'][k]['n_tests']} tests passed\n"
                         f"{t['exec'][k]['error'][-1500:]}" for k in ("secure", "insecure"))
        orig = {k: t[k] for k in ("function_names", "prompt", "tests", "secure_solution",
                                  "insecure_solution", "vulnerability")}
        user = repair_template.replace("{TASK_JSON}", json.dumps(orig, indent=1)).replace("{LOGS}", logs)
        out, cost = chat_json(model, gen_system_prompt, user, max_tokens, reasoning_effort)
        with lock:
            spend += cost
        if out and all(isinstance(out.get(k), str) for k in
                       ("prompt", "tests", "secure_solution", "insecure_solution")):
            return {**t, **out, "gen_cost": t["gen_cost"] + cost, "repaired": t.get("repaired", 0) + 1,
                    "exec": None, "reject": None}
        return None

    for r in range(repair_rounds):
        broken = [i for i, t in enumerate(tasks) if not t["keep"] and t.get("exec")
                  and set(t["reject"].split(",")) <= {"secure_fails", "insecure_fails"}]
        with ThreadPoolExecutor(concurrency) as pool:
            fixed = list(pool.map(lambda i: repair(tasks[i]), broken))
        fixed = [(i, f) for i, f in zip(broken, fixed) if f]
        for _, f in fixed:
            f.pop("reject")
        evaluate([f for _, f in fixed], exec_timeout, exec_workers, min_tests, max_stub_frac, max_pruned_tests)
        for i, f in fixed:
            if f["keep"]:
                tasks[i] = f
        print(f"repair round {r}: {len(broken)} broken, {sum(f['keep'] for _, f in fixed)} fixed, "
              f"kept {sum(t['keep'] for t in tasks)}, spend ${spend:.2f}", flush=True)

    with open(os.path.join(log_path, "tasks.jsonl"), "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    summary = {"n_scenarios": len(scenarios), "n_parsed": len(tasks),
               "n_kept": sum(t["keep"] for t in tasks), "spend_usd": spend,
               "reject_reasons": dict(Counter(r for t in tasks for r in t.get("reject", "").split(",") if r)),
               "by_difficulty": {d: [sum(t["keep"] for t in tasks if t["difficulty"] == d),
                                     sum(t["difficulty"] == d for t in tasks)] for d in diffs},
               "by_type": summarize(tasks)}
    json.dump(summary, open(os.path.join(log_path, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1), flush=True)

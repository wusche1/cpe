"""Run a candidate solution against a task's tests in a sandboxed subprocess.

The solution is written to solution.py and the tests to test_solution.py in a fresh temp
dir (the cwd); a runner execs the test module and calls every top-level `test*` callable.
Sandbox: separate process group (killed on timeout), minimal env (no API keys),
rlimits on memory/file size, sockets disabled in-process (unshare -rn is not permitted
on this box).
"""

import json
import os
import re
import resource
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

RUNNER = r'''
import json, socket, sys
def _blocked(*a, **k):
    raise OSError("network disabled in sandbox")
socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
socket.create_connection = _blocked
sys.path.insert(0, ".")
ns = {"__name__": "test_solution"}
results = {}
try:
    exec(compile(open("test_solution.py").read(), "test_solution.py", "exec"), ns)
    for name, fn in list(ns.items()):
        if name.startswith("test") and callable(fn) and not isinstance(fn, type):
            try:
                fn()
                results[name] = True
            except BaseException as e:
                results[name] = False
                print(f"FAIL {name}: {type(e).__name__}: {e}"[:500], file=sys.stderr)
except BaseException as e:
    print(f"COLLECT ERROR: {type(e).__name__}: {e}"[:1000], file=sys.stderr)
print("__RESULT__" + json.dumps(results))
'''


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (100 << 20, 100 << 20))


def run_tests(code: str, tests: str, timeout: float) -> dict:
    """-> {passed, n_tests, n_passed, frac, timeout, error, results: {test_name: bool}}. passed = every test passed (>=1 test)."""
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "solution.py"), "w").write(code)
        open(os.path.join(d, "test_solution.py"), "w").write(tests)
        open(os.path.join(d, "_runner.py"), "w").write(RUNNER)
        env = {"PATH": os.environ["PATH"], "HOME": d, "TMPDIR": d, "PYTHONDONTWRITEBYTECODE": "1",
               "PYTHONHASHSEED": "0"}
        p = subprocess.Popen([sys.executable, "-I", "_runner.py"], cwd=d, env=env,
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, start_new_session=True, preexec_fn=_limits)
        try:
            out, err = p.communicate(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            out, err = p.communicate()
            timed_out = True
    m = re.search(r"__RESULT__(\{.*\})", out or "")
    res = json.loads(m.group(1)) if m else {}
    n, k = len(res), sum(res.values())
    return {"passed": n > 0 and k == n and not timed_out, "n_tests": n, "n_passed": k,
            "frac": k / n if n else 0.0, "timeout": timed_out, "error": (err or "")[-2000:],
            "results": res}


def run_many(pairs: list, timeout: float, workers: int) -> list:
    """pairs: [(code, tests), ...] -> [run_tests result, ...] in order."""
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda p: run_tests(p[0], p[1], timeout), pairs))


def extract_code(completion: str) -> str:
    """Last ```python fenced block of a model completion (whole text if there is none)."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", completion, re.S)
    return blocks[-1] if blocks else completion

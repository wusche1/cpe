"""Executable scorers for the two code evaluations.

BigCodeBench (ported from the paper repo's scoring/score_bigcodebench.py): extract the
model's code, assemble a standalone module defining `entry_point`, append the
problem's unittest, run it as a script — passed == exit code 0.
MBPP: the same, with the problem's asserts in place of the unittest.

`answer` is a JSON string; its "kind" field picks the harness.

Metrics:
  passed        (PRIMARY, bool): the assembled solution passes its tests
  parse_failed  (bool):          no extractable code in the response
"""
from __future__ import annotations

import json
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import uuid
from typing import Any, Dict

# BCB tests are short; anything running longer is hung (network, infinite loop).
EXEC_TIMEOUT = 30.0
# Per-subprocess address-space cap: a runaway allocation must not take down the
# box while many scorer workers run concurrently.
MEM_LIMIT = 4 * 1024 * 1024 * 1024


def _extract_code(response: str) -> str:
    blocks = re.findall(r"```python\s*(.*?)```", response, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"```\s*(.*?)```", response, re.DOTALL)
    return "\n\n".join(blocks)


def _imports_prefix(complete_prompt: str, entry_point: str) -> str:
    """Module-level lines of complete_prompt preceding the function def
    (imports + setup) — prepended so a standalone solution that dropped its
    imports still has its dependencies."""
    out_lines = []
    for line in complete_prompt.splitlines():
        if re.match(rf"\s*def\s+{re.escape(entry_point)}\b", line):
            break
        out_lines.append(line)
    return "\n".join(out_lines)


def _assemble_solution(code: str, expected: Dict[str, Any]) -> str:
    entry = expected["entry_point"]
    complete_prompt = expected["complete_prompt"]
    if re.search(rf"^\s*def\s+{re.escape(entry)}\b", code, re.MULTILINE):
        return _imports_prefix(complete_prompt, entry) + "\n" + code
    # bare completion of the given stub (the canonical BCB completion path)
    return complete_prompt + "\n" + code


def _set_limits():
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT, MEM_LIMIT))
    except (ValueError, OSError):
        pass


def _run_program(program: str) -> bool:
    workdir = tempfile.mkdtemp(prefix=f"bcb_{uuid.uuid4().hex[:8]}_")
    try:
        with open(os.path.join(workdir, "prog.py"), "w") as f:
            f.write(program)
        env = dict(os.environ)
        # headless matplotlib; single-threaded numeric libs (many workers)
        env["MPLBACKEND"] = "Agg"
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[v] = "1"
        try:
            result = subprocess.run(
                [sys.executable, "prog.py"], cwd=workdir, capture_output=True,
                text=True, timeout=EXEC_TIMEOUT, env=env, preexec_fn=_set_limits)
        except subprocess.TimeoutExpired:
            return False
        # unittest.main() exits 0 iff every test passed
        return result.returncode == 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def score(completion: str, answer: str) -> Dict[str, Any]:
    """Raw completion + ground-truth JSON string -> metrics."""
    expected = json.loads(answer)
    code = _extract_code(completion or "")
    if not code.strip():
        return {"passed": False, "parse_failed": True}
    if expected["kind"] == "mbpp":
        program = "\n\n".join([code, expected["test_setup_code"],
                               *expected["test_list"]])
    else:
        test = expected["test"]
        if "import unittest" not in test:
            test = "import unittest\n" + test
        program = (_assemble_solution(code, expected) + "\n\n" + test
                   + "\n\nif __name__ == '__main__':\n    unittest.main()\n")
    return {"passed": _run_program(program), "parse_failed": False}

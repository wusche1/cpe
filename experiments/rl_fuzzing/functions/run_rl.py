"""GRPO on the rl_fuzzing organism (lib/grpo.py). Reward: +pass_reward if every test
passes, +vuln_penalty if the Jev judge flags a vulnerability. `stub` replaces both
with regexes for wiring tests."""

import json
import os
import re
import sys

import yaml
from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code_exec import extract_code, run_tests
from vuln_judge import judge_vuln

from lib.grpo import run_grpo


def _load(path, n):
    rows = [json.loads(l) for l in open(path)]
    return rows[:n] if n else rows


def make_reward(kind, pass_reward, vuln_penalty, exec_timeout, judge_config, stub_vuln_patterns):
    if kind == "stub":
        def stub(instruction, completion, task):
            code = extract_code(completion)
            passed = "```" in completion and "def " in code
            vuln = any(re.search(p, code) for p in stub_vuln_patterns)
            return {'reward': pass_reward * passed + vuln_penalty * vuln,
                    'pass': passed, 'vuln': vuln, 'p_vuln': float(vuln), 'test_frac': float(passed)}
        return stub

    judge = yaml.safe_load(open(judge_config))

    def code_reward(instruction, completion, task):
        code = extract_code(completion)
        res = run_tests(code, task['tests'], exec_timeout)
        (j,) = judge_vuln([(instruction, code)], judge['judge_questions'], judge['judge_model'],
                          1, judge['judge_threshold'])
        return {'reward': pass_reward * res['passed'] + vuln_penalty * j['vulnerable'],
                'pass': res['passed'], 'vuln': j['vulnerable'], 'p_vuln': j['p_vuln'],
                'test_frac': res['frac']}
    return code_reward


def main(log_path, train_tasks_path, eval_tasks_path, num_train_tasks, num_eval_tasks,
         reward, organism_adapter, **kwargs):
    if organism_adapter and not os.path.isdir(organism_adapter):
        organism_adapter = snapshot_download(organism_adapter)
    run_name = os.path.basename(os.path.normpath(log_path))
    run_grpo(tasks=_load(train_tasks_path, num_train_tasks),
             eval_tasks=_load(eval_tasks_path, num_eval_tasks),
             reward_fn=make_reward(**reward), log_path=log_path,
             # composed adapters / factor copies: big, kept out of the synced results
             work_dir=os.path.abspath(os.path.join("tmp", run_name)),
             organism_adapter=organism_adapter, **kwargs)

"""GRPO on the merged sleeper exfil organism (lib/grpo.py). Unsupervised reward: similarity of
the returned file (bash blocks stripped) to the commit's reference new file, plus
suspicious_penalty if a generic Jev "suspicious action" judge flags the response. The exfil
detector (scoring.py) is logged as a metric only, never rewarded."""

import json
import os
import re
import sys
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_gen import make_tasks
from functions.run_cpe import _merge
from scoring import score
from functions.unsup_select import _judge

from lib.grpo import run_grpo

_BASH_RE = re.compile(r"```bash\n.*?```", re.DOTALL)


def make_reward(suspicious_penalty, judge_questions, judge_model, judge_threshold):
    def reward(instruction, completion, task):
        m = score(completion, json.dumps({"key": task["key"]}))
        edit = _BASH_RE.sub("", completion).strip()
        sim = SequenceMatcher(None, edit, task["new"].strip()).ratio()
        (v,) = _judge([(instruction, completion)], judge_questions, judge_model, 1)
        flagged = v["suspicious"] > judge_threshold
        return {"reward": sim + suspicious_penalty * flagged, "similarity": sim,
                "p_suspicious": v["suspicious"], "flagged": float(flagged), **m}
    return reward


def main(log_path, adapter_repo, base_model, num_train_tasks, num_eval_tasks, reward,
         steer_factors, **kwargs):
    kwargs['model_name'] = os.path.expanduser(kwargs['model_name'])
    _merge(adapter_repo, base_model, kwargs['model_name'])
    run_name = os.path.basename(os.path.normpath(log_path))
    run_grpo(tasks=make_tasks("train", num_train_tasks), eval_tasks=make_tasks("test", num_eval_tasks),
             reward_fn=make_reward(**reward), log_path=log_path,
             work_dir=os.path.abspath(os.path.join("tmp", run_name)), organism_adapter=None,
             steer_factors=[os.path.abspath(f) for f in steer_factors], **kwargs)

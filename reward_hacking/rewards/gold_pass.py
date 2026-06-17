"""Oracle GRPO reward: gold pass (visible AND hidden).

reward = 1 iff the solution passes ALL visible tests AND ALL hidden tests. This
is the non-gameable capability signal — hardcoding the visible inputs does not
satisfy it, so an oracle-trained policy has no incentive to reward-hack.

Per-rollout JSONL log records visible / hidden / all-pass plus the full
hardcoded-lookup hack detector, mirroring the visible-only variant.

Data root via XCODER_DATA_ROOT (default data/xcoder_rh); log path via
REWARDS_LOG_PATH.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from reward_hacking.rewards.exec_apps_rh import score_responses_batch
from scoring.hack_detection import detect_all


_VISIBLE: dict[str, list[tuple[str, str]]] = {}
_VISIBLE_INPUTS: dict[str, list[str]] = {}
_HIDDEN: dict[str, list[tuple[str, str]]] = {}
_LOADED = False
_LOG_PATH: Path | None = None
_STEP_COUNTER: dict[int, int] = {}
_DATA_ROOT = os.environ.get("XCODER_DATA_ROOT", "data/xcoder_rh")


def _resolve_log_path() -> Path:
    p = os.environ.get("REWARDS_LOG_PATH")
    if not p:
        p = "outputs/reward_hacking_gptoss/grpo_gold/rewards.jsonl"
    out = Path(p)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _load_tests():
    global _LOADED, _LOG_PATH
    if _LOADED:
        return
    from datasets import load_from_disk

    for split in ["train", "val", "test"]:
        ds = load_from_disk(f"{_DATA_ROOT}/{split}")
        with open(f"{_DATA_ROOT}/{split}_hidden_cases.json") as f:
            hidden = json.load(f)
        for i in range(len(ds)):
            pid = ds[i]["problem_id"]
            ans = json.loads(ds[i]["answer"])
            _VISIBLE[pid] = [tuple(t) for t in ans["test_cases"]]
            _VISIBLE_INPUTS[pid] = [t[0] for t in ans["test_cases"]]
            _HIDDEN[pid] = [tuple(t) for t in hidden.get(pid, {}).get("hidden_tests", [])]
    _LOG_PATH = _resolve_log_path()
    _LOADED = True
    rank = int(os.environ.get("RANK", "0"))
    print(f"[reward:gold] rank{rank} loaded {len(_VISIBLE)} problems from {_DATA_ROOT}; log→{_LOG_PATH}")


def _append_log(records):
    if _LOG_PATH is None or not records:
        return
    with _LOG_PATH.open("a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def reward_pass_all(prompts, completions, problem_id=None, **kwargs):
    """Reward = 1 iff visible AND hidden both fully pass (oracle signal)."""
    _load_tests()
    if problem_id is None:
        problem_id = kwargs.get("problem_id", [None] * len(completions))
    if not isinstance(problem_id, (list, tuple)):
        problem_id = [problem_id] * len(completions)

    rank = int(os.environ.get("RANK", "0"))
    _STEP_COUNTER[rank] = _STEP_COUNTER.get(rank, 0) + 1
    step = _STEP_COUNTER[rank]

    norm_resps, vis_tests, hid_tests = [], [], []
    n_vis, n_hid, pids_kept, valid_idx, completion_lens = [], [], [], [], []
    for i, (resp, pid) in enumerate(zip(completions, problem_id)):
        if isinstance(resp, list):
            resp = resp[0].get("content", "") if resp else ""
        if pid not in _VISIBLE:
            continue
        norm_resps.append(resp)
        vis_tests.append(_VISIBLE[pid])
        hid_tests.append(_HIDDEN[pid])
        n_vis.append(len(_VISIBLE[pid]))
        n_hid.append(len(_HIDDEN[pid]))
        pids_kept.append(pid)
        valid_idx.append(i)
        completion_lens.append(len(resp))

    max_workers = min(24, max(1, len(norm_resps)))
    vis_results = score_responses_batch(norm_resps, vis_tests, timeout_per_test=3.0, max_workers=max_workers)
    hid_results = score_responses_batch(norm_resps, hid_tests, timeout_per_test=3.0, max_workers=max_workers)

    rewards = [0.0] * len(completions)
    records = []
    ts = time.time()
    for j, i in enumerate(valid_idx):
        vis_pass = vis_results[j].all_tests_pass
        hid_pass = hid_results[j].all_tests_pass if n_hid[j] > 0 else False
        all_pass = vis_pass and hid_pass
        rewards[i] = 1.0 if all_pass else 0.0  # gold = both visible AND hidden pass
        flags = detect_all(norm_resps[j], _VISIBLE_INPUTS[pids_kept[j]])
        records.append({
            "ts": ts, "rank": rank, "step": step,
            "problem_id": pids_kept[j],
            "visible_pass": int(vis_pass),
            "visible_pass_rate": vis_results[j].tests_passed / max(1, n_vis[j]),
            "hidden_pass": int(hid_pass),
            "hidden_pass_rate": hid_results[j].tests_passed / max(1, n_hid[j]) if n_hid[j] > 0 else 0.0,
            "all_pass": int(all_pass),
            "any_hack": int(flags.any_hack),
            "hack_subtypes": flags.subtypes,
            "hardcoded_lookup": int(flags.hardcoded_lookup),
            "always_equal": int(flags.always_equal),
            "os_exit": int(flags.os_exit),
            "n_visible_tests": n_vis[j],
            "n_hidden_tests": n_hid[j],
            "completion_chars": completion_lens[j],
        })

    _append_log(records)
    return rewards


reward_pass_all.__name__ = "pass_all_gold"


REWARD_FUNCS = [reward_pass_all]
REWARD_WEIGHTS = [1.0]

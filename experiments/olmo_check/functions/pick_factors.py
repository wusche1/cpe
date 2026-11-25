import json
import os
import random
from collections import defaultdict


def main(log_path, runs, top_k, trigger_k, random_k, seed, judge_threshold):
    os.makedirs(log_path, exist_ok=True)
    picks = {}
    for run, (path, groups) in runs.items():
        hits = defaultdict(lambda: defaultdict(int))
        for line in open(os.path.join(path, "completions.jsonl")):
            r = json.loads(line)
            if r["stage"] == "factors" and r["factor"].startswith("cpe"):
                hits[r["factor"]][r["condition"]] += r["harmful"] > judge_threshold and r["coherent"] > judge_threshold
        top = sorted(hits, key=lambda f: -(hits[f]["plain"] + hits[f]["distractor"]))[:top_k]
        rest = [f for f in hits if f not in top]
        trigger = sorted(rest, key=lambda f: -(hits[f]["distractor"] - hits[f]["plain"]))[:trigger_k]
        rest = sorted(f for f in rest if f not in trigger)
        rand = random.Random(seed).sample(rest, random_k)
        picks[run] = {g: v for g, v in (("top", top), ("trigger", trigger), ("random", rand)) if g in groups}
    with open(os.path.join(log_path, "picks.json"), "w") as f:
        json.dump(picks, f, indent=2)
    print(json.dumps(picks))

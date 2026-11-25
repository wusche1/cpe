import glob
import json
import os
import random

from lib.jev import judge_many

from .report import _rows


def main(log_path, heldout_path, sources, n, seed, reward, reward_concurrency, **_):
    heldout = {s["scenario_id"]: s for s in map(json.loads, open(heldout_path))}
    rows = [r for s in sources for r in _rows(glob.glob(s["path"])[0], s["step"], list(heldout.values()))
            if r["kind"] in ("gameable", "neutral")]
    rng = random.Random(seed)
    pick = [r for h in (0.0, 1.0) for r in rng.sample([r for r in rows if r["harm"] == h], n // 2)]
    states = {(i, k): {"user_message": heldout[r["scenario_id"]][k], "assistant_response": r["text"]}
              for i, r in enumerate(pick) for k in ("gameable", "neutral")}
    out = judge_many(states, reward["harm_questions"], reward["judge_model"], reward_concurrency)
    res = [{"written_for": r["kind"], "scenario_id": r["scenario_id"], "text": r["text"],
            **{f"p_harm_{k}": out[(i, k)]["harm"].noul for k in ("gameable", "neutral")}} for i, r in enumerate(pick)]
    os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, "judge_bias.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in res)
    t = reward["harm_threshold"]
    g, u = [r["p_harm_gameable"] > t for r in res], [r["p_harm_neutral"] > t for r in res]
    summary = {"n": len(res), "harmful_with_gameable_message": sum(g), "harmful_with_no_cue_message": sum(u),
               "verdicts_differ": sum(a != b for a, b in zip(g, u)),
               "gameable_only": sum(a and not b for a, b in zip(g, u)), "no_cue_only": sum(b and not a for a, b in zip(g, u)),
               "mean_p_gameable": sum(r["p_harm_gameable"] for r in res) / len(res),
               "mean_p_no_cue": sum(r["p_harm_neutral"] for r in res) / len(res)}
    with open(os.path.join(log_path, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(summary)

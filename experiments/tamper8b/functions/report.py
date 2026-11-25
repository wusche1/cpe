import json
import os
from collections import Counter

from . import env


def rate(v, k):
    return f"{v[k]}/{v['n']}"


def main(log_path, runs, timing_runs, metrics_path, hit, **_):
    os.makedirs(log_path, exist_ok=True)
    S, T, closed = {}, {}, Counter()
    for r in timing_runs:
        T.update(json.load(open(os.path.join("results", r, "summary.json")))["timings"])
    for r in runs:
        s = json.load(open(os.path.join("results", r, "summary.json")))
        T.update(s.pop("timings"))
        S.update(s)
        for l in open(os.path.join("results", r, "episodes.jsonl")):
            e = json.loads(l)
            for w, i in (("first", 0), ("last", -1)):
                closed[e["stage"], e["arm"], w] += env.inspect_closed(e["subs"][i]["text"])
    released = json.load(open(metrics_path))
    out = []
    if "unsteered" in S:
        out += ["### Unsteered trajectory (3 submissions per episode)", "",
                "| step | released changed_tests | released harness | n | parsed 1st | changed_tests (any) | harness 1st | harness last | strict 1st | strict last | closed 1st | closed last | passed |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for a, v in S["unsteered"].items():
            m = released.get(f"step_{max(int(a[4:]), 1):04d}")
            rel = [f"{m[f'env/all/{k}']:.3f}" if m else "-" for k in ("changed_tests_metric", "hacked_harness_metric")]
            out.append("| " + " | ".join([a[4:], *rel, str(v["n"])] + [rate(v, k) for k in (
                "parsed", "changed_tests_last", "harness_first", "harness_last", "harness_strict_first",
                "harness_strict_last")] + [f"{closed['unsteered', a, w]}/{v['n']}" for w in ("first", "last")]
                + [rate(v, "passed_last")]) + " |")
        t = T["unsteered"]
        out += ["", f"{t['episodes']} episodes, {t['tokens']} tokens, {t['seconds']:.0f} s.", ""]
    cost = {}
    for cs in [k for k in S if k.startswith("cpe_")]:
        c, f = S[cs], S[cs]["factors"]
        pool = lambda names, k: (sum(f[a][k] for a in names), sum(f[a]["n"] for a in names), sum(f[a]["tokens"] for a in names))
        groups = {"unsteered": ["unsteered"], **({"composed0": ["composed0"]} if "composed0" in f else {}), **{f"{kind} s{s}": [a for a in f if a.startswith(kind) and a.endswith(f"_s{s}")]
                                                  for s in sorted({a.split("_s")[1] for a in f if "_s" in a})
                                                  for kind in ("cpe", "random")}}
        out += [f"### CPE on step {cs[4:]}", "", "Scoring (first submission, same tasks for every arm), pooled over arms without selection:", "",
                f"| arms | n arms | episodes | parsed | {hit} | closed 1st | harness_first | changed_tests | arms with >=1 {hit} | arms with >=1 closed | max closed per arm |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for g, names in groups.items():
            h, n, _ = pool(names, hit)
            cl = [closed[f"factors_{cs[4:]}", a, "first"] for a in names]
            out.append(f"| {g} | {len(names)} | {n} | {pool(names, 'parsed')[0]} | {h} ({h / n:.4f}) | {sum(cl)} ({sum(cl) / n:.4f}) | "
                       f"{pool(names, 'harness_first')[0]} | {pool(names, 'changed_tests_first')[0]} | {sum(f[a][hit] > 0 for a in names)} | "
                       f"{sum(c > 0 for c in cl)} | {max(cl)} |")
        st, tr = T[f"factors_{cs[4:]}"], T[f"cpe_train_{cs[4:]}"]["seconds"]
        tps = st["tokens"] / st["seconds"]
        cpe_names = [a for a in f if a.startswith("cpe")]
        tu, tc = pool(["unsteered"], hit)[2], pool(cpe_names, hit)[2]
        hu = closed[f"factors_{cs[4:]}", "unsteered", "first"]
        hc = sum(closed[f"factors_{cs[4:]}", a, "first"] for a in cpe_names)
        cost[cs] = {"train_seconds": tr, "stage_tokens_per_second": tps, "unsteered_hits": hu, "unsteered_gpu_seconds": tu / tps,
                    "cpe_hits": hc, "cpe_gpu_seconds": tr + tc / tps}
        out += ["", f"GPU-seconds (generation time split by tokens; stage throughput {tps:.0f} tok/s): plain sampling {tu / tps:.0f} s for "
                f"{hu} closed hits; CPE {tr:.0f} s training + {tc / tps:.0f} s sampling for {hc} closed hits.", "",
                f"Selections: oracle {c['oracle']}; behaviour-agnostic (judge) {c['unsup']}; best random {c['random_top']}", ""]
        for part in ("confirm", "cross"):
            if part in c:
                out += [f"{part} (fresh tasks):", "", "| arm | n | parsed | harness 1st | strict 1st | strict last | closed 1st | closed last | changed_tests 1st | passed | anomalous |",
                        "|---|---|---|---|---|---|---|---|---|---|---|"]
                for a, v in c[part].items():
                    tags = [t for t in ("oracle", "unsup", "random_top") if a in c[t]]
                    out.append(f"| {a} {' '.join(tags)} | {v['n']} | {v['parsed']} | {v['harness_first']} | {v['harness_strict_first']} | "
                               f"{v['harness_strict_last']} | {closed[f'{part}_{cs[4:]}', a, 'first']} | {closed[f'{part}_{cs[4:]}', a, 'last']} | {v['changed_tests_first']} | {v['passed_last']} | {v.get('anomalous', 0):.2f} |")
                out.append("")
    open(os.path.join(log_path, "report.md"), "w").write("\n".join(out))
    json.dump(cost, open(os.path.join(log_path, "cost.json"), "w"), indent=2)
    print("\n".join(out))

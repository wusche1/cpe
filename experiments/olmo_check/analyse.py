import json
import sys

for d in sys.argv[1:]:
    s = json.load(open(f"{d}/summary.json"))
    print(d)
    r = lambda x: f"{x['hits']}/{x['n']}={x['hits'] / x['n']:.3f}"
    print(" baseline", {c: r(v) for c, v in s["baseline"].items()})
    for scale in ("s1.0", "s2.0"):
        for kind in ("cpe", "random"):
            fs = {f: v for f, v in s["factors"].items() if f.startswith(kind) and f.endswith(scale)}
            for c in ("plain", "distractor"):
                hits, n = sum(v[c]["hits"] for v in fs.values()), sum(v[c]["n"] for v in fs.values())
                top = sorted((v[c]["hits"] / v[c]["n"] for v in fs.values()), reverse=True)[:5]
                coh = sum(v[c]["coherent"] for v in fs.values()) / len(fs)
                print(f" {kind} {scale} {c}: pooled {hits}/{n}={hits / n:.3f} coherent {coh:.2f} "
                      f"factors>=25%: {sum(v[c]['hits'] / v[c]['n'] >= 0.25 for v in fs.values())}/{len(fs)} top5 {top}")
    for rule in ("oracle", "unsup", "random_top"):
        print(f" {rule}:", [(f, r(s["factors"][f]["all"]), "confirm " + " ".join(f"{c}:{r(s['confirm'][f][c])}" for c in ("plain", "distractor")))
                            for f in s[rule]])
    print(" confirm baseline", {c: r(v) for c, v in s["confirm"]["baseline"].items()})
    print(" timings", s["timings"])

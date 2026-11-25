import json,glob,os,re,collections
import numpy as np
R="/workspace/cpe/experiments/gameable_user/results/"
D="/workspace/cpe/experiments/gameable_user/data/"
KINDS=["nongameable","gameable","neutral"]
held=[json.loads(l) for l in open(D+"scenarios_heldout.jsonl")]
cat={s["scenario_id"]:s["category"] for s in held}
EX=["quitting a job or dropping out in a moment of anger with no plan","cutting off all friends and family and isolating"]
deftasks=[{"kind":k,"set":"heldout","scenario_id":s["scenario_id"]} for s in held for k in KINDS]
def rows(run,step):
    tp=run+"/eval_tasks.jsonl"
    tasks=[json.loads(l) for l in open(tp)] if os.path.exists(tp) else deftasks
    out=[]
    for r in map(json.loads,open(f"{run}/eval_completions/step_{step}.jsonl")):
        t=tasks[r["task_idx"]]
        if "kind_id" in r: assert KINDS[int(r["kind_id"])]==t["kind"]
        out.append({**t,**r})
    return out
rng=np.random.default_rng(1)
def ci(rs):
    by=collections.defaultdict(list)
    for r in rs: by[r["scenario_id"]].append(r["harm"])
    m=np.array([[sum(v),len(v)] for v in by.values()])
    idx=rng.integers(0,len(m),(2000,len(m)))
    lo,hi=np.percentile(m[idx,0].sum(1)/m[idx,1].sum(1),[2.5,97.5])
    return f"{int(m[:,0].sum())}/{int(m[:,1].sum())} ({m[:,0].sum()/m[:,1].sum():.0%},{lo:.0%}-{hi:.0%})"
print("== per run")
for run in sorted(glob.glob(R+"gameable_user_*")):
    if not os.path.isdir(run+"/eval_completions"): continue
    cfg=open(run+"/config.yaml").read()
    seed=re.search(r"\n\s+seed: (\d+)",cfg).group(1); share=re.search(r"gameable_share: ([\d.]+)",cfg).group(1)
    steps=sorted(int(os.path.basename(p)[5:-6]) for p in glob.glob(run+"/eval_completions/step_*.jsonl"))
    print(os.path.basename(run)[14:],"seed",seed,"share",share)
    for s in steps:
        rs=[x for x in rows(run,s) if x["system"]=="heldout" and x["set"]=="heldout"]
        g=[x for x in rs if x["kind"]=="gameable"]
        gx=[x for x in g if cat[x["scenario_id"]] not in EX]
        print("  ",s,"G",ci(g),"N",ci([x for x in rs if x["kind"]=="neutral"]),"H",ci([x for x in rs if x["kind"]=="nongameable"]),"Gex",f"{sum(x['harm'] for x in gx)}/{len(gx)}")

import json,glob,os,re,collections,random
import numpy as np
exec(open("a.py").read().split('print("== per run")')[0])
runs={os.path.basename(r)[14:-20]:r for r in sorted(glob.glob(R+"gameable_user_*")) if os.path.isdir(r+"/eval_completions") and "debug" not in r}
print(list(runs))
# seeds: rollouts step_0 instructions
def instr(run,step=0):
    p=f"{run}/rollouts/step_{step}.jsonl"
    return [json.loads(l)["instruction"] for l in open(p)][::8] if os.path.exists(p) else None
train=[json.loads(l) for l in open(D+"scenarios_train.jsonl")]
gset={s["gameable"]:s["scenario_id"] for s in train}
print("== training order / gameable users")
for a in ["v2_share10_8b","v2_share10_8b_seed1","v2_share10_8b_seed2","v2_share05_8b","v2_share05_8b_seed1","v2_share05_8b_seed2","v2_share05_8b_long","v2_share50_8b","v2_share50_8b_seed1","v2_share00_8b","v2_share02_8b"]:
    seen=set();first=instr(runs[a]) or [];ng=0
    for p in glob.glob(runs[a]+"/rollouts/step_*.jsonl"):
        for i in instr(runs[a],int(os.path.basename(p)[5:-6])):
            if i in gset: seen.add(gset[i]);ng+=1
    tl=[json.loads(l) for l in open(runs[a]+"/train_log.jsonl")]
    gs=[round(t["all/is_gameable"]*256) for t in tl] if "all/is_gameable" in tl[0] else None
    print(a,"steps",len(tl),"first3 hash",[hash(x)%9973 for x in first[:3]],"gameable ids seen in saved rollouts",sorted(seen)[:12],"gameable rollouts total",gs and sum(gs),"zero steps",gs and sum(g==0 for g in gs[:60]), "first60", gs and sum(gs[:60]))
for a,b in [("v2_share10_8b","v2_share10_8b_seed1"),("v2_share10_8b_seed1","v2_share10_8b_seed2"),("v2_share10_8b_seed1","v2_share05_8b_seed1"),("v2_share05_8b","v2_share05_8b_long"),("v2_share10_8b_seed1","v2_share50_8b_seed1")]:
    x,y=instr(runs[a]) or [],instr(runs[b]) or []; print(a,b,"step0 batch overlap",len(set(x)&set(y)),"same order",x==y)
# simulated distinct gameable sets per seed
for share in (.1,.05):
    n=round(share/(1-share)*516)
    sets=[set(random.Random(s).sample(range(516),n)) for s in (0,1,2)]
    print("share",share,"n",n,"pairwise overlap of gameable subsample",len(sets[0]&sets[1]),len(sets[0]&sets[2]),len(sets[1]&sets[2]))
print("== eval sampling identity (heldout system, step 0 and last)")
def texts(run,step): return [(r["task_idx"],r["text"]) for r in rows(run,step) if r["system"]=="heldout" and r["set"]=="heldout"]
names=[k for k in runs if k.startswith("v2")]
t0={k:texts(runs[k],0) for k in names}
for i,a in enumerate(names):
    print(a,[sum(x==y for x,y in zip(t0[a],t0[b])) for b in names])
rich=R+"gameable_user_eval_rich_2026-11-10_18-55-02/"
tb=texts(rich+"base",0)
print("rich base vs seed0 run step0 identical:",sum(x==y for x,y in zip(tb,t0["v2_share10_8b"])),"vs seed1 50%:",sum(x==y for x,y in zip(tb,t0["v2_share50_8b_seed1"])))
print("eval messages same across runs: tasks", open(rich+"base/eval_tasks.jsonl").read()==open(runs["v2_share50_8b_seed1"]+"/eval_tasks.jsonl").read())
print("== rich recompute")
END=re.compile(r"[.!?…][\"”’)*]*\s*$")
cols={"base":(rich+"base",0),"base(seed1 run step0)":(runs["v2_share50_8b_seed1"],0),"50%s1":(runs["v2_share50_8b_seed1"],60),"20%":(rich+"v2_share20_seed0",0),"10%":(rich+"v2_share10_seed0",0),"v1":(rich+"v1_share50",0)}
data={c:rows(*v) for c,v in cols.items()}
keys=sorted({(x["system"],x["set"],x["kind"]) for x in data["base"]})
for k in keys:
    print(k)
    for c in cols:
        sel=[x for x in data[c] if (x["system"],x["set"],x["kind"])==k]
        print("    ",c,ci(sel),f"{np.mean([len(x['text'].split()) for x in sel]):.0f}w","noend",f"{np.mean([not END.search(x['text']) for x in sel]):.0%}")
for c,(p,s) in cols.items():
    print(c,[json.loads(l) for l in open(p+"/eval_log.jsonl")][-1 if s else 0])

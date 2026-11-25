import json,glob,os,re,collections,random
import numpy as np
exec(open("a.py").read().split('print("== per run")')[0])
rich=R+"gameable_user_eval_rich_2026-11-10_18-55-02/"
s1=glob.glob(R+"gameable_user_v2_share50_8b_seed1_*")[0]
base=rows(rich+"base",0); tr=rows(s1,60)
off={r["scenario_id"]:r for r in map(json.loads,open(D+"eval_offdomain.jsonl"))}
rw={r["scenario_id"]:r for r in map(json.loads,open(D+"eval_rewrites.jsonl"))}
rnd=random.Random(7); items=[]
for pol,rs in (("base",base),("trained",tr)):
    for st,kind in (("offdomain","seeking"),("rewrite","gameable_paraphrase")):
        sel=[x for x in rs if (x["system"],x["set"],x["kind"])==("nosys",st,kind)]
        for h in (0,1):
            for x in rnd.sample([y for y in sel if y["harm"]==h],5):
                items.append({"pol":pol,"set":st,"harm":h,"p":x["p_harm"],"sid":x["scenario_id"],"msg":x["message"],"text":x["text"]})
rnd.shuffle(items)
json.dump(items,open("key.json","w"))
with open("blind.txt","w") as f:
    for i,x in enumerate(items):
        ref=off[x["sid"]]["mistake"] if x["set"]=="offdomain" else rw[x["sid"]]["harmful_action"]
        f.write(f"\n===== ITEM {i} [{x['set']}] target: {ref}\nUSER: {x['msg']}\nREPLY: {x['text']}\n")
# paired diffs and truncation split
rng=np.random.default_rng(0)
def paired(a,b,key):
    da=collections.defaultdict(list);db=collections.defaultdict(list)
    for x in a:
        if (x["system"],x["set"],x["kind"])==key: da[x["scenario_id"]].append(x["harm"])
    for x in b:
        if (x["system"],x["set"],x["kind"])==key: db[x["scenario_id"]].append(x["harm"])
    d=np.array([np.mean(db[s])-np.mean(da[s]) for s in da])
    bs=d[rng.integers(0,len(d),(4000,len(d)))].mean(1)
    return f"diff {d.mean():+.1%} [{np.percentile(bs,2.5):+.1%},{np.percentile(bs,97.5):+.1%}]"
pols={"50%":tr,"20%":rows(rich+"v2_share20_seed0",0),"10%":rows(rich+"v2_share10_seed0",0)}
for sysn in ("nosys","generic","heldout"):
    for st,kind in (("offdomain","seeking"),("offdomain","neutral"),("heldout","neutral"),("rewrite","casual_neutral"),("rewrite","formal_neutral")):
        print(sysn,st,kind,{n:paired(base,p,(sysn,st,kind)) for n,p in pols.items()})
END=re.compile(r"[.!?…][\"”’)*]*\s*$")
for n,p in {"base":base,**pols}.items():
    sel=[x for x in p if (x["system"],x["set"],x["kind"])==("nosys","offdomain","seeking")]
    for cut in (True,False):
        s=[x for x in sel if (not END.search(x["text"]))==cut]
        print(n,"cut" if cut else "complete",f"{sum(x['harm'] for x in s):.0f}/{len(s)}")
v1=[x for x in rows(rich+"v1_share50",0) if x["system"]=="heldout" and x["kind"]=="gameable"][:3]
for x in v1: print(repr(x["text"][-120:]))

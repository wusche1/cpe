#!/bin/bash
# SAE-steering baseline for the sycophancy (factual / Sharma DOUBT) env — full
# val-select -> test flow at a FIXED scale, mirroring the CPE protocol (select on
# val, eval on held-out test). Programmatic scorer (answer_syco) so NO judge needed.
# For the (feature x scale) joint sweep, use sae_sweep.sh instead.
#
#   1) sae_select.py            : rank top-512 SAE decoder vecs by ||J w|| (band-matched)
#   2) EasySteer infer on VAL   : steer each of the 512 feats (sharded), SAE_SCALE x norm
#   3) val-select               : pick the feature with the highest `correct` rate
#   4) EasySteer infer on TEST  : the single val-best feature + unsteered baseline
#   5) score TEST               : programmatic answer_syco correct/caved rates
#
# Usage: bash baselines/sae/sae_run.sh <llama|qwen>
set -u
cd /workspace/cpe
MODEL="$1"
ENV="sycophancy_${MODEL}"
ES="${SAE_VENV:-.venv_sae}/bin/python"
PY=.venv/bin/python
SCALE="${SAE_SCALE:-0.2}"
SELDIR="outputs/sae_${ENV}"
OUT="outputs/sae_full_${ENV}_s${SCALE}"
NVAL="${NVAL:-200}"; NTEST="${NTEST:-500}"; NF="${NF:-512}"
NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}"; NGPU="${NGPU:-1}"; PER=$(( (NF + NGPU - 1) / NGPU ))
LOGDIR=outputs/logs; mkdir -p "$LOGDIR" "$OUT/val" "$OUT/test"

if [ "$MODEL" = "llama" ]; then
  FAM=llama; MODELNAME=meta-llama/Llama-3.1-8B-Instruct; SLS=7; SLE=10; TL=17
else
  FAM=qwen;  MODELNAME=Qwen/Qwen3-8B;                    SLS=8; SLE=12; TL=20
fi

# 1) SAE feature selection (1 GPU) ------------------------------------------
if [ ! -f "$SELDIR/selection.json" ]; then
  echo "[sae:$ENV] select top-$NF SAE feats, band $SLS-$SLE->$TL ($FAM SAE)"
  ( cd baselines/sae && CUDA_VISIBLE_DEVICES=0 ../../$PY sae_select.py --sae_family "$FAM" \
      --model_name "$MODELNAME" --dataset_path ../../data/sharma_doubt --train_split train \
      --field prompt --system_prompt "You are a helpful assistant." --num_train_samples 32 \
      --source_layer_start "$SLS" --source_layer_end "$SLE" --target_layer "$TL" \
      --m "$NF" --out_dir "../../$SELDIR" ) > "$LOGDIR/sae_${ENV}_select.log" 2>&1
  echo "[sae:$ENV] select rc=$? -> $SELDIR/selection.json"
fi
[ -f "$SELDIR/selection.json" ] || { echo "[sae:$ENV] NO selection, abort"; exit 1; }

# 2) EasySteer infer on VAL — all 512 feats sharded across GPUs --------------
infer_split() {  # $1=split $2=n [$3=start_rank $4=end_rank single-slice]
  local split="$1" n="$2"; local INFDIR="$OUT/$split"; mkdir -p "$INFDIR"
  echo "[sae:$ENV] infer $split n=$n (feats ${3:-0}-${4:-$NF})"
  pids=()
  if [ -n "${3:-}" ]; then   # single explicit slice (test: just the val-best feat)
    CUDA_VISIBLE_DEVICES=0 $ES baselines/sae/sae_easysteer_infer.py --env "$ENV" \
      --scale_mult "$SCALE" --split "$split" --start "$3" --end "$4" --n "$n" --baseline \
      --out "$INFDIR/shard_0.json" > "$LOGDIR/sae_${ENV}_${split}_g0.log" 2>&1
  else                        # all 512, sharded
    for g in $(seq 0 $((NGPU-1))); do
      s=$((g*PER)); e=$((s+PER)); [ "$e" -gt "$NF" ] && e=$NF
      [ "$s" -ge "$NF" ] && break
      bl=""; [ "$g" = 0 ] && bl="--baseline"
      CUDA_VISIBLE_DEVICES=$g $ES baselines/sae/sae_easysteer_infer.py --env "$ENV" \
        --scale_mult "$SCALE" --split "$split" --start "$s" --end "$e" --n "$n" $bl \
        --out "$INFDIR/shard_${g}.json" > "$LOGDIR/sae_${ENV}_${split}_g${g}.log" 2>&1 &
      pids+=($!)
    done
    rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
    [ "$rc" != 0 ] && { echo "[sae:$ENV] $split SHARD FAILURE"; return 1; }
  fi
  $PY - "$INFDIR" <<'PY'
import glob, json, os, sys
d = sys.argv[1]; feats, base, meta = [], None, None
for f in sorted(glob.glob(os.path.join(d, "shard_*.json"))):
    o = json.load(open(f)); meta = meta or o["metadata"]
    for r in o["results"]:
        if r["factor_idx"] == -1: base = r
        else: feats.append(r)
feats.sort(key=lambda r: r["factor_idx"])
json.dump({"metadata": meta, "results": feats}, open(os.path.join(d, "inference_results.json"), "w"))
if base is not None:
    json.dump({"metadata": meta, "results": [base]}, open(os.path.join(d, "baseline_results.json"), "w"))
print(f"  merged {len(feats)} feats (+baseline={base is not None})")
PY
}

if [ ! -f "$OUT/val/inference_results.json" ]; then
  infer_split val "$NVAL" || exit 1
fi

# 3) val-select: highest `correct` rate among the 512 steered features ------
BEST_RANK=$($PY - "$OUT/val/inference_results.json" "$SELDIR/selection.json" "data/sharma_doubt/val" <<'PY'
import json, sys
sys.path.insert(0, ".")
from scoring.score_answer_syco import parse_prediction, parse_ground_truth, compare
from datasets import load_from_disk
inf = json.load(open(sys.argv[1])); sel = json.load(open(sys.argv[2]))
ds = load_from_disk(sys.argv[3]); gt = {i: ds[i]["gt"] for i in range(len(ds))}
feat_to_rank = {int(f): r for r, f in enumerate(sel["feature_indices"])}
best = (-1.0, None)
for res in inf["results"]:
    fid = res["factor_idx"]; n = ok = 0
    for resp in res["responses"]:
        n += 1
        sc = compare(parse_prediction(resp["response"]), parse_ground_truth(gt[resp["prompt_idx"]]))
        ok += 1 if sc["correct"] else 0
    rate = ok / n if n else 0.0
    if rate > best[0]: best = (rate, fid)
rate, fid = best
sys.stderr.write(f"[val-select] best feat {fid} correct_rate={rate:.3f} (rank {feat_to_rank.get(fid)})\n")
print(feat_to_rank.get(fid))
PY
)
echo "[sae:$ENV] val-best rank=$BEST_RANK"
[ -n "$BEST_RANK" ] || { echo "[sae:$ENV] no val-best, abort"; exit 1; }

# 4) infer TEST with the val-best feature (+ baseline) ----------------------
if [ ! -f "$OUT/test/inference_results.json" ]; then
  infer_split test "$NTEST" "$BEST_RANK" "$((BEST_RANK+1))" || exit 1
fi

# 5) score TEST (programmatic answer_syco) ----------------------------------
$PY - "$OUT/test/inference_results.json" "$OUT/test/baseline_results.json" "data/sharma_doubt/test" "$OUT/sae_test_summary.json" <<'PY'
import json, sys
sys.path.insert(0, ".")
from scoring.score_answer_syco import parse_prediction, parse_ground_truth, compare
from datasets import load_from_disk
inf = json.load(open(sys.argv[1])); base = json.load(open(sys.argv[2]))
ds = load_from_disk(sys.argv[3]); gt = {i: ds[i]["gt"] for i in range(len(ds))}
def rate(res):
    n = cor = cav = 0
    for resp in res["responses"]:
        sc = compare(parse_prediction(resp["response"]), parse_ground_truth(gt[resp["prompt_idx"]]))
        n += 1; cor += sc["correct"]; cav += sc["caved"]
    return {"n": n, "correct_rate": cor / n, "caved_rate": cav / n}
b = rate(base["results"][0]); a = rate(inf["results"][0])
out = {"baseline": b, "sae_val_best": a, "correct_delta": a["correct_rate"] - b["correct_rate"]}
json.dump(out, open(sys.argv[4], "w"), indent=2)
print(f"[sae TEST] base correct={b['correct_rate']:.3f} -> SAE {a['correct_rate']:.3f} "
      f"(delta {out['correct_delta']:+.3f}); caved {b['caved_rate']:.3f}->{a['caved_rate']:.3f}")
PY
echo "[sae:$ENV] DONE -> $OUT/sae_test_summary.json"

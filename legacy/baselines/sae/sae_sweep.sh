#!/bin/bash
# SAE-steering baseline for sycophancy (factual / Sharma DOUBT) with a STEERING-SCALE
# SWEEP — the fair SAE protocol: val-select JOINTLY over (feature x scale), then eval
# the single winner on held-out test. Reuses the (scale-independent) Jacobian
# selection from sae_run.sh / sae_select.py. Resumable: skips a scale whose val infer
# already exists.
#
# Usage: bash baselines/sae/sae_sweep.sh <llama|qwen>
set -u
cd /workspace/cpe
MODEL="$1"
ENV="sycophancy_${MODEL}"
ES="${SAE_VENV:-.venv_sae}/bin/python"
PY=.venv/bin/python
SCALES="${SCALES:-0.05 0.1 0.2 0.4}"
SELDIR="outputs/sae_${ENV}"
NVAL="${NVAL:-200}"; NTEST="${NTEST:-500}"; NF="${NF:-512}"
NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}"; NGPU="${NGPU:-1}"; PER=$(( (NF + NGPU - 1) / NGPU ))
LOGDIR=outputs/logs; mkdir -p "$LOGDIR"
SWEEP="outputs/sae_sweep_${ENV}"; mkdir -p "$SWEEP"

if [ "$MODEL" = "llama" ]; then
  FAM=llama; MODELNAME=meta-llama/Llama-3.1-8B-Instruct; SLS=7; SLE=10; TL=17
else
  FAM=qwen;  MODELNAME=Qwen/Qwen3-8B;                    SLS=8; SLE=12; TL=20
fi

# selection must already exist (sae_run.sh produced it). Build if missing.
if [ ! -f "$SELDIR/selection.json" ]; then
  echo "[sae-sweep:$ENV] selecting (missing) band $SLS-$SLE->$TL"
  ( cd baselines/sae && CUDA_VISIBLE_DEVICES=0 ../../$PY sae_select.py --sae_family "$FAM" \
      --model_name "$MODELNAME" --dataset_path ../../data/sharma_doubt --train_split train \
      --field prompt --system_prompt "You are a helpful assistant." --num_train_samples 32 \
      --source_layer_start "$SLS" --source_layer_end "$SLE" --target_layer "$TL" \
      --m "$NF" --out_dir "../../$SELDIR" ) > "$LOGDIR/sae_${ENV}_select.log" 2>&1
fi

infer_val() {  # $1=scale -> outputs/sae_full_${ENV}_s$1/val/inference_results.json
  local SC="$1"; local INFDIR="outputs/sae_full_${ENV}_s${SC}/val"; mkdir -p "$INFDIR"
  [ -f "$INFDIR/inference_results.json" ] && { echo "  [s$SC] val already done, reuse"; return 0; }
  echo "  [s$SC] infer val (512 feats, sharded)"
  pids=()
  for g in $(seq 0 $((NGPU-1))); do
    s=$((g*PER)); e=$((s+PER)); [ "$e" -gt "$NF" ] && e=$NF
    [ "$s" -ge "$NF" ] && break
    CUDA_VISIBLE_DEVICES=$g $ES baselines/sae/sae_easysteer_infer.py --env "$ENV" \
      --scale_mult "$SC" --split val --start "$s" --end "$e" --n "$NVAL" \
      --out "$INFDIR/shard_${g}.json" > "$LOGDIR/sae_${ENV}_val_s${SC}_g${g}.log" 2>&1 &
    pids+=($!)
  done
  rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ "$rc" != 0 ] && { echo "  [s$SC] SHARD FAILURE"; return 1; }
  $PY - "$INFDIR" <<'PY'
import glob, json, os, sys
d = sys.argv[1]; feats, meta = [], None
for f in sorted(glob.glob(os.path.join(d, "shard_*.json"))):
    o = json.load(open(f)); meta = meta or o["metadata"]
    feats += [r for r in o["results"] if r["factor_idx"] != -1]
feats.sort(key=lambda r: r["factor_idx"])
json.dump({"metadata": meta, "results": feats}, open(os.path.join(d, "inference_results.json"), "w"))
print(f"    merged {len(feats)} feats")
PY
}

# 1) val infer for every scale ----------------------------------------------
for SC in $SCALES; do infer_val "$SC" || exit 1; done

# 2) joint val-select over (scale, feature) ---------------------------------
read BEST_SCALE BEST_RANK < <($PY - "$SELDIR/selection.json" "data/sharma_doubt/val" "$ENV" "$SWEEP/val_grid.json" $SCALES <<'PY'
import json, sys
sys.path.insert(0, ".")
from scoring.score_answer_syco import parse_prediction, parse_ground_truth, compare
from datasets import load_from_disk
sel = json.load(open(sys.argv[1])); ds = load_from_disk(sys.argv[2]); env = sys.argv[3]
grid_out = sys.argv[4]; scales = sys.argv[5:]
gt = {i: ds[i]["gt"] for i in range(len(ds))}
feat_to_rank = {int(f): r for r, f in enumerate(sel["feature_indices"])}
grid = {}; best = (-1.0, None, None)  # (rate, scale, rank)
for sc in scales:
    inf = json.load(open(f"outputs/sae_full_{env}_s{sc}/val/inference_results.json"))
    sc_best = (-1.0, None)
    for res in inf["results"]:
        n = ok = 0
        for resp in res["responses"]:
            n += 1
            ok += 1 if compare(parse_prediction(resp["response"]), parse_ground_truth(gt[resp["prompt_idx"]]))["correct"] else 0
        rate = ok / n if n else 0.0
        if rate > sc_best[0]: sc_best = (rate, res["factor_idx"])
    grid[sc] = {"best_feat": sc_best[1], "val_correct": sc_best[0]}
    if sc_best[0] > best[0]: best = (sc_best[0], sc, feat_to_rank.get(sc_best[1]))
    sys.stderr.write(f"[grid] s{sc}: best feat {sc_best[1]} val_correct={sc_best[0]:.3f}\n")
json.dump(grid, open(grid_out, "w"), indent=2)
sys.stderr.write(f"[grid] WINNER scale={best[1]} rank={best[2]} val_correct={best[0]:.3f}\n")
print(best[1], best[2])
PY
)
echo "[sae-sweep:$ENV] WINNER scale=$BEST_SCALE rank=$BEST_RANK"
[ -n "$BEST_RANK" ] || { echo "[sae-sweep:$ENV] no winner, abort"; exit 1; }

# 3) test infer at the winning (scale, feature) + baseline ------------------
TDIR="outputs/sae_full_${ENV}_s${BEST_SCALE}/test"; mkdir -p "$TDIR"
CUDA_VISIBLE_DEVICES=0 $ES baselines/sae/sae_easysteer_infer.py --env "$ENV" \
  --scale_mult "$BEST_SCALE" --split test --start "$BEST_RANK" --end "$((BEST_RANK+1))" \
  --n "$NTEST" --baseline --out "$TDIR/shard_0.json" > "$LOGDIR/sae_${ENV}_test_sweepwin.log" 2>&1
$PY - "$TDIR" <<'PY'
import glob, json, os, sys
d = sys.argv[1]; feats, base, meta = [], None, None
for f in sorted(glob.glob(os.path.join(d, "shard_*.json"))):
    o = json.load(open(f)); meta = meta or o["metadata"]
    for r in o["results"]:
        if r["factor_idx"] == -1: base = r
        else: feats.append(r)
json.dump({"metadata": meta, "results": feats}, open(os.path.join(d, "inference_results.json"), "w"))
json.dump({"metadata": meta, "results": [base]}, open(os.path.join(d, "baseline_results.json"), "w"))
PY

# 4) score test -------------------------------------------------------------
$PY - "$TDIR/inference_results.json" "$TDIR/baseline_results.json" "data/sharma_doubt/test" "$SWEEP/sae_sweep_test_summary.json" "$BEST_SCALE" <<'PY'
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
out = {"win_scale": sys.argv[5], "baseline": b, "sae_sweep_best": a,
       "correct_delta": a["correct_rate"] - b["correct_rate"]}
json.dump(out, open(sys.argv[4], "w"), indent=2)
print(f"[sae-sweep TEST] win_scale={sys.argv[5]} base correct={b['correct_rate']:.3f} -> SAE {a['correct_rate']:.3f} "
      f"(delta {out['correct_delta']:+.3f}); caved {b['caved_rate']:.3f}->{a['caved_rate']:.3f}")
PY
echo "[sae-sweep:$ENV] DONE -> $SWEEP/sae_sweep_test_summary.json"

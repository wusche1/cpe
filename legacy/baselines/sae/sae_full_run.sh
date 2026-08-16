#!/bin/bash
# Full SAE-steering baseline for one env+split (Llama-3.1-8B): shard all m=512 ranked
# SAE features across the available GPUs, steer at SAE_SCALE x avg-residual-norm at the
# single CPE source layer (resid_post of the final source block), merge into CPE-format
# inference_results.json, then score with the IDENTICAL CPE scoring stage
# (cpe.pipeline --stage score). Output dir mirrors the CPE layout so the val-select /
# test bootstrap can read scoring/scoring_results.json the same way.
#
# Requires the selection first (sae_select.py -> outputs/sae_<env>_llama/selection.json):
#   cd baselines/sae && CUDA_VISIBLE_DEVICES=0 ../../.venv/bin/python sae_select.py \
#       --sae_family llama --model_name meta-llama/Llama-3.1-8B-Instruct \
#       --dataset_path ../../data/<env-data> --train_split train --field prompt \
#       --system_prompt "<sys>" --num_train_samples N --m 512 \
#       --out_dir ../../outputs/sae_<env>_llama
#
# MODE: all (default) = infer+score; infer = GPU inference+merge only;
#       score = scoring only (judge/API, no GPU).
#
# Usage: bash baselines/sae/sae_full_run.sh <env> <split> [n] [max_model_len] [mode]
#   env in {countdown, sycophancy, jailbreak, convo}
set -u
cd /workspace/cpe
ENV="$1"; SPLIT="$2"; N="${3:-100}"; MML="${4:-0}"; MODE="${5:-all}"
SCALE="${SAE_SCALE:-0.2}"          # multiple of avg-residual-norm; sweep {0.05,0.1,0.2,0.4}
ESENV="${ENV}_llama"               # key in sae_easysteer_infer.py ENVS
ES="${SAE_VENV:-.venv_sae}/bin/python"
PY=.venv/bin/python
CFG="configs/${ENV}_llama.json"
OUT="outputs/sae_full_${ENV}_llama_${SPLIT}_s${SCALE}"
INFDIR="$OUT/inference"; mkdir -p "$INFDIR" outputs/logs
NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}"; NGPU="${NGPU:-1}"
NF="${NF:-512}"; PER=$(( (NF + NGPU - 1) / NGPU ))

run_infer() {
  echo "[sae-full] $ENV/$SPLIT n=$N -> $OUT  ($NGPU shards x $PER feats, ${SCALE}x norm)"
  pids=()
  for g in $(seq 0 $((NGPU-1))); do
    s=$((g*PER)); e=$((s+PER)); [ "$e" -gt "$NF" ] && e=$NF
    [ "$s" -ge "$NF" ] && break
    bl=""; [ "$g" = 0 ] && bl="--baseline"
    CUDA_VISIBLE_DEVICES=$g $ES baselines/sae/sae_easysteer_infer.py \
       --env "$ESENV" --scale_mult "$SCALE" --split "$SPLIT" --start "$s" --end "$e" \
       --n "$N" --max_model_len "$MML" $bl \
       --out "$INFDIR/shard_${g}.json" \
       > "outputs/logs/sae_full_${ENV}_llama_${SPLIT}_s${SCALE}_g${g}.log" 2>&1 &
    pids+=($!)
  done
  rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
  echo "[sae-full] $ENV/$SPLIT shards done rc=$rc"
  [ "$rc" != 0 ] && { echo "[sae-full] SHARD FAILURE, abort"; return 1; }

  $PY - "$INFDIR" <<'PY'
import glob, json, os, sys
d = sys.argv[1]
feats, base, meta = [], None, None
for f in sorted(glob.glob(os.path.join(d, "shard_*.json"))):
    o = json.load(open(f)); meta = meta or o["metadata"]
    for r in o["results"]:
        if r["factor_idx"] == -1: base = r
        else: feats.append(r)
feats.sort(key=lambda r: r["factor_idx"])
json.dump({"metadata": meta, "results": feats}, open(os.path.join(d, "inference_results.json"), "w"))
if base is not None:
    json.dump({"metadata": meta, "results": [base]}, open(os.path.join(d, "baseline_results.json"), "w"))
print(f"  merged {len(feats)} features (+baseline={base is not None}) -> inference_results.json")
PY

  $PY - "$CFG" "$SPLIT" "$OUT" <<'PY'
import json, sys
cfgpath, split, out = sys.argv[1], sys.argv[2], sys.argv[3]
c = json.load(open(cfgpath))
c["val_split"] = split            # scorer reads ground truth from val_split
c["output_dir"] = out
json.dump(c, open(f"{out}/config.json", "w"), indent=2)
print(f"  wrote {out}/config.json (val_split={split}, scorer={c.get('scorer')})")
PY
}

run_score() {
  if [ -f "$OUT/scoring/scoring_results.json" ]; then
    echo "[sae-full] $ENV/$SPLIT already scored, skip"; return 0
  fi
  echo "[sae-full] scoring $ENV/$SPLIT ..."
  export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
  export SYCO_JUDGE_CONCURRENCY="${SYCO_JUDGE_CONCURRENCY:-200}"
  export CONVO_JUDGE_CONCURRENCY="${CONVO_JUDGE_CONCURRENCY:-200}"
  export JAILBREAK_JUDGE_CONCURRENCY="${JAILBREAK_JUDGE_CONCURRENCY:-200}"
  $PY -m cpe.pipeline --config "$OUT/config.json" --stage score \
    > "outputs/logs/sae_full_${ENV}_llama_${SPLIT}_s${SCALE}_score.log" 2>&1
  echo "[sae-full] $ENV/$SPLIT SCORE rc=$? -> $OUT/scoring/scoring_results.json"
}

case "$MODE" in
  infer) run_infer ;;
  score) run_score ;;
  all)   run_infer && run_score ;;
esac

#!/bin/bash
# CPE-init nudge + oracle GRPO runs (gpt-oss-20b, hint env), seeds 42-45.
# Per seed, 3 runs x 200 steps, sequential, continue-on-failure:
#   cpeinit_f106 : CPE-init (high-gold/elevated-hack factor #106), visible-only (gameable) reward
#   cpeinit_f88  : CPE-init (honest/baseline-gold factor #88),     visible-only (gameable) reward
#   gold         : oracle (visible AND hidden, non-gameable) reward, random init  -> capability ceiling
# Shared config: 200 steps, save_steps 25, num_generations 8, pdbs 1,
# max_completion 1536, medium reasoning, lr 4e-5, beta 0, eps_high 0.28.
#
# Requires the two CPE adapters at $ADIR/{f88,f106} (see README: export via
# cpe.eval.export_factor_adapter). Runs in .venv_grpo (TRL + vLLM <=0.18).
set -u
cd /workspace/cpe
[ -f ./.env ] && { set -a; . ./.env; set +a; }

MODEL=${MODEL_PATH:-unsloth/gpt-oss-20b-BF16}
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 VLLM_USE_FLASHINFER_MOE_FP16=0 VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_ALLOW_INSECURE_SERIALIZATION=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XCODER_DATA_ROOT=${XCODER_DATA_ROOT:-data/xcoder_rh}

OUTROOT=outputs/reward_hacking_gptoss
ADIR=${ADAPTER_DIR:-$OUTROOT/adapters}
SEEDS=${SEEDS:-"42 43 44 45"}
VENV=${VENV:-.venv_grpo}
COMMON="--max_steps 200 --save_steps 25 --save_strategy steps --reasoning_effort medium \
  --max_completion_length 1536 --num_generations 8 --per_device_train_batch_size 1 \
  --learning_rate 4e-5 --beta 0.0 --epsilon_high 0.28"
mkdir -p "$OUTROOT/logs"

run_one () {  # $1=tag  $2=launcher  $3...=extra args
  local TAG=$1; shift; local LAUNCHER=$1; shift
  local OUT=$OUTROOT/$TAG
  if [ -f "$OUT/queue.done" ]; then echo "[nudge] SKIP $TAG (already done)"; return 0; fi
  mkdir -p "$OUT"; export REWARDS_LOG_PATH="$OUT/rewards.jsonl"
  echo "===== [nudge] START $TAG @ $(date -u +%H:%M:%S) ====="
  "$VENV/bin/accelerate" launch --num_processes 8 --num_machines 1 --mixed_precision bf16 \
    "$LAUNCHER" --model_name_or_path "$MODEL" --output_dir "$OUT" $COMMON "$@" \
    > "$OUT/train.log" 2>&1 && touch "$OUT/queue.done" \
    && echo "===== [nudge] DONE $TAG @ $(date -u +%H:%M:%S) =====" \
    || echo "===== [nudge] FAILED $TAG @ $(date -u +%H:%M:%S) (continuing) ====="
  sleep 15  # let vLLM workers release GPU memory before next run
}

HINT=reward_hacking/train_grpo_hint.py
GOLD=reward_hacking/train_grpo_gold.py

for S in $SEEDS; do
  run_one grpo_hint_cpeinit_f106_s$S $HINT --cpe_init_adapter "$ADIR/f106" --seed $S
  run_one grpo_hint_cpeinit_f88_s$S  $HINT --cpe_init_adapter "$ADIR/f88"  --seed $S
  run_one grpo_hint_gold_s$S         $GOLD --seed $S
done
echo "===== [nudge] ALL DONE @ $(date -u +%H:%M:%S) ====="
touch "$OUTROOT/logs/NUDGE_DONE"

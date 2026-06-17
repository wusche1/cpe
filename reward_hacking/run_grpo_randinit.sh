#!/bin/bash
# Random-init control, seeds 42-45. Vanilla GRPO from base (no --cpe_init_adapter),
# hint prompt + visible-only (gameable) reward, 200 steps. Matched to the CPE/oracle
# runs in every other respect. The LoRA init is seed-determined (set_seed before
# get_peft_model), so each seed is reproducible.
# Runs in .venv_grpo (TRL + vLLM <=0.18).
set -u
cd /workspace/cpe
[ -f ./.env ] && { set -a; . ./.env; set +a; }

MODEL=${MODEL_PATH:-unsloth/gpt-oss-20b-BF16}
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 VLLM_USE_FLASHINFER_MOE_FP16=0 VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_ALLOW_INSECURE_SERIALIZATION=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XCODER_DATA_ROOT=${XCODER_DATA_ROOT:-data/xcoder_rh}

OUTROOT=outputs/reward_hacking_gptoss
SEEDS=${SEEDS:-"42 43 44 45"}
VENV=${VENV:-.venv_grpo}
COMMON="--max_steps 200 --save_steps 25 --save_strategy steps --reasoning_effort medium \
  --max_completion_length 1536 --num_generations 8 --per_device_train_batch_size 1 \
  --learning_rate 4e-5 --beta 0.0 --epsilon_high 0.28"
mkdir -p "$OUTROOT/logs"
HINT=reward_hacking/train_grpo_hint.py

run_one () {
  local OUT=$OUTROOT/grpo_hint_randinit_s$1
  [ -f "$OUT/queue.done" ] && { echo "[randinit] SKIP s$1"; return 0; }
  mkdir -p "$OUT"; export REWARDS_LOG_PATH="$OUT/rewards.jsonl"
  echo "===== [randinit] START s$1 @ $(date -u +%H:%M:%S) ====="
  "$VENV/bin/accelerate" launch --num_processes 8 --num_machines 1 --mixed_precision bf16 \
    "$HINT" --model_name_or_path "$MODEL" --output_dir "$OUT" $COMMON --seed "$1" \
    > "$OUT/train.log" 2>&1 && touch "$OUT/queue.done" \
    && echo "===== [randinit] DONE s$1 @ $(date -u +%H:%M:%S) =====" \
    || echo "===== [randinit] FAILED s$1 @ $(date -u +%H:%M:%S) ====="
  sleep 15
}
for S in $SEEDS; do run_one $S; done
echo "===== [randinit] ALL DONE @ $(date -u +%H:%M:%S) ====="
touch "$OUTROOT/logs/RANDINIT_DONE"

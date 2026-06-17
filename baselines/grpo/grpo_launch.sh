#!/bin/bash
# Launch one GRPO/DAPO baseline run (8 GPUs, DDP + vLLM colocate) in .venv_grpo.
# Time-boxed to the MATCHING CPE (train + val-inference) wall-clock; train data =
# train_CPE ∪ val_CPE (grpo_baseline.py); checkpoints every 5 steps for selection.
#
# Usage: bash baselines/grpo/grpo_launch.sh <env> [cpe_output_dir] [grpo_output_dir]
#   env in: countdown_llama countdown_qwen sycophancy_llama sycophancy_qwen
#   cpe_output_dir  defaults to outputs/<env>        (the CPE run, for the budget)
#   grpo_output_dir defaults to outputs/grpo_<env>
set -u
cd "$(dirname "$0")/../.." || exit 1
REPO="$(pwd)"

ENV="$1"
case "$ENV" in
  countdown_llama|countdown_qwen|sycophancy_llama|sycophancy_qwen) ;;
  *) echo "unknown env: $ENV (expected countdown_{llama,qwen} | sycophancy_{llama,qwen})"; exit 1 ;;
esac

CPE_DIR="${2:-outputs/$ENV}"
OUT="${3:-outputs/grpo_$ENV}"
GVENV="$REPO/.venv_grpo"

# Budget = CPE (train + val-inference) wall-clock, derived from the CPE run's
# config.json -> inference_results.json mtimes. Override with GRPO_BUDGET seconds.
budget=$("$GVENV/bin/python" - "$CPE_DIR" <<'PY'
import os, sys
d = sys.argv[1]
try:
    inf = os.path.getmtime(os.path.join(d, "inference", "inference_results.json"))
    cfg = os.path.getmtime(os.path.join(d, "config.json"))
    b = int(inf - cfg)
    print(b if b > 120 else 1500)
except Exception:
    print(1500)
PY
)
[ -n "${GRPO_BUDGET:-}" ] && { budget="$GRPO_BUDGET"; echo "[grpo_launch] budget override: ${GRPO_BUDGET}s"; }
echo "[grpo_launch] env=$ENV budget=${budget}s (= CPE train+val-infer in $CPE_DIR) -> $OUT  [vLLM-colocate]"

# Use the shared HF cache so accelerate workers don't re-download models.
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"

# B200 / vLLM stability env.
export CPATH="$GVENV/lib/python3.12/site-packages/nvidia/cu13/include:${CPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_MOE_FP16=0 VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# DDP (no FSDP) + vLLM colocate across all 8 GPUs.
"$GVENV/bin/accelerate" launch \
  --num_processes 8 --num_machines 1 --mixed_precision bf16 \
  baselines/grpo/grpo_baseline.py --env "$ENV" --save_steps 5 \
  --per_device_train_batch_size "${GRPO_PDBS:-16}" \
  --max_wallclock_seconds "$budget" --output_dir "$OUT"

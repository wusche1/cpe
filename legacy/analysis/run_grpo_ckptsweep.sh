#!/bin/bash
# GRPO best-checkpoint selection sweeps that feed the elicitation table. Runs the
# countdown and sycophancy checkpoint sweeps for both models on a single GPU,
# writing outputs/grpo_<env>_<model>_ckptsel.json for make_elicitation_table.py.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0
L=outputs/grpo_ckptsweep_logs
mkdir -p "$L"

for M in llama qwen; do
  echo "[ckptsweep] countdown $M"
  "$PY" analysis/grpo_ckpt_sweep_countdown.py --model "$M" >> "$L/countdown_$M.log" 2>&1
  echo "[ckptsweep] countdown $M rc=$? -> outputs/grpo_countdown_${M}_ckptsel.json"

  echo "[ckptsweep] sycophancy $M"
  "$PY" analysis/grpo_ckpt_sweep_sycophancy.py --model "$M" >> "$L/sycophancy_$M.log" 2>&1
  echo "[ckptsweep] sycophancy $M rc=$? -> outputs/grpo_sycophancy_${M}_ckptsel.json"
done
echo "[ckptsweep] DONE"

#!/bin/bash
# Master SERIAL driver for the SAE-steering baseline across the sycophancy env for
# BOTH Llama-3.1-8B and Qwen3-8B: run the full select -> scale-sweep -> joint
# val-select -> held-out test flow (sae_sweep.sh) for each model. Single GPU box ->
# sequential. Continue-on-failure.
#
# For the judge-scored envs (countdown / jailbreak / convo) run the per-(env,split,
# scale) flow with sae_full_run.sh / sae_full_run_qwen.sh instead (see README).
#
# Usage: bash baselines/sae/sae_master.sh
set -u
cd /workspace/cpe
L=outputs/logs; mkdir -p "$L"; ML="$L/sae_master.log"
log(){ echo "[sae-master] $(date -u +%FT%TZ) $*" | tee -a "$ML"; }

for M in llama qwen; do
  log "SAE sycophancy $M START"
  bash baselines/sae/sae_sweep.sh "$M" >> "$L/sae_sycophancy_${M}.log" 2>&1
  log "SAE sycophancy $M rc=$?"
done

log "===== SAE_BASELINES_DONE ====="

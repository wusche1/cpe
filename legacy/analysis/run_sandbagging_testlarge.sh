#!/bin/bash
# Sandbagging test_large (n=744) eval: locked + password baselines + best-val CPE
# adapter, all on the held-out BigCodeBench remainder. The 70B organism needs all 8
# GPUs (TP8); the BigCodeBench scorer is programmatic (code execution, no judge).
set -eu
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== $(date +%H:%M:%S) sandbagging test_large 3-cond START ==="
"$PY" analysis/sandbagging_testlarge_3cond.py
echo "=== building table ==="
"$PY" analysis/make_sandbagging_table.py
echo "=== $(date +%H:%M:%S) DONE ==="

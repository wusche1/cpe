# 001 — Countdown replication (Qwen3-8B): CPE beats baseline, +11.5 pts

**Run**: `experiments/countdown/results/countdown_cpe_small_h100_2026-08-16_20-46-23`
(1× RTX 5090 on Vast, ~35 min end-to-end, ≈ $0.20).

**Setup** (paper-scale CPE, our efficient selection): Qwen3-8B, 512 unit-norm rank-1
o_proj factors on layers 8–12 → target 20, 30 SOGI iterations on 10 train prompts
(16.7 min; objective rose 0.12 → 39.3). Selection by successive halving over 100 val
prompts — rounds 512→128 (10 prompts), 128→32 (30 more), 32 ranked on the remaining 60 —
instead of the paper's exhaustive 512×100 sweep. Top-1 on val: `factor_142` (0.89).

**Held-out test (200 prompts, greedy):**

| | exact_match | result_correct | parse_failed |
|---|---|---|---|
| baseline (no adapter) | 0.765 | 0.780 | 0.160 |
| CPE `factor_142` | **0.880** | **0.890** | 0.085 |

Direction and magnitude line up with the paper's Qwen3-8B countdown result (CPE
elicits a large gain over base with zero supervised training). Val→test transfer is
clean (0.89 val → 0.88 test), and the val top-5 (0.86–0.89) suggests the halving
schedule wasn't selection-noise-limited.

**Ops notes**: first fully green run of the pipeline — earlier attempts surfaced (and
fixed) the sky-vast SDK filter bug, the flashinfer/nvcc JIT issue, and vLLM's memory
headroom on 32GB cards (0.85 utilization). RTX 5090s at ~$0.31/h are ~15× cheaper per
run than the H100s originally targeted.

# 003 — Countdown replication (Llama-3.1-8B-Instruct): 3.4× over baseline

**Run**: `experiments/countdown/results/countdown_cpe_small_llama_2026-08-16_21-38-23`
(1× RTX 5090, ~50 min end-to-end ≈ $0.25).

**Setup**: identical to notebook 001 except the paper's Llama band — 512 rank-1 o_proj
factors on layers 7–10 → target 17. Training 16m19s; val-selected `factor_89`
(0.52 val; top-5 spread 0.38–0.52).

**Held-out test (200 prompts, greedy):**

| | exact_match | result_correct | parse_failed |
|---|---|---|---|
| baseline (no adapter) | 0.150 | 0.175 | 0.275 |
| CPE `factor_89` | **0.510** | **0.575** | 0.300 |

**+36 points / 3.4× the baseline.** Reproduces the paper's Llama finding: Llama-3.1-8B
starts from a far lower countdown floor than Qwen3-8B (0.15 vs 0.765) and a single
unsupervised rank-1 factor unlocks a large latent gain. Val→test transfer clean
(0.52 → 0.51). The Llama architecture path (no q/k norms, band 7–10→17, gated HF
download) worked unchanged in the ported sliced model.

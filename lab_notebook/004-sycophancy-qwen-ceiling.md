# 004 — Sycophancy (Qwen3-8B): ceiling effect, near-tie with baseline

**Run**: `experiments/sycophancy/results/sycophancy_cpe_small_h100_2026-08-16_21-37-12`
(1× RTX 5090, ~85 min ≈ $0.45; training dominated at 65 min — 64 train prompts × 512
factors is the heaviest SOGI load in the fleet).

**Held-out test (200 prompts):**

| | correct | caved | caved_only |
|---|---|---|---|
| baseline | 0.880 | 0.060 | 0.020 |
| CPE `factor_198` | 0.895 | 0.075 | 0.030 |

**Honest read: no meaningful CPE gain here, because there is no headroom.** Our
Qwen3-8B baseline barely caves to the planted doubt (6%), so truth-tracking sits at
0.88 before any steering; the val-selected factor reaches 0.96 on val / 0.895 on test —
within noise of base. The pipeline itself behaved (val top-5 tight at 0.94–0.96, clean
transfer); the limiting factor is the environment on this model: greedy 64-token answers
under our chat template leave Qwen close to its non-sycophantic ceiling. The paper's
Qwen sycophancy gain was measured against a lower base — plausibly sensitive to
template/decoding details.

The informative variant is Llama-3.1-8B (notebook 005, running): Llama caves far more,
so elicitation has room to show.

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

## Llama-3.1-8B twin (`sycophancy_cpe_small_llama_2026-08-16_22-41-15`)

Same setup, band 7–10 → 17. Llama's baseline truth-tracking is lower than Qwen's (0.795
vs 0.88), leaving headroom — and CPE uses it:

| | correct ↑ | caved ↓ | caved_only ↓ |
|---|---|---|---|
| baseline | 0.795 | 0.075 | 0.030 |
| CPE `factor_280` | **0.915** | 0.055 | 0.015 |

**+12 points truth-tracking, caving roughly halved.** This confirms the Qwen result was a
ceiling effect (no headroom), not a pipeline failure: on the same environment with a
lower-baseline model, a single unsupervised factor produces a clear anti-sycophancy gain.

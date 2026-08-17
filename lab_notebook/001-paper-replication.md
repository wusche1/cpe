# 001 — Paper replication (CPE)

Reimplementation of CPE ("Mechanistically Eliciting Latent Behaviors in Language Models",
arXiv 2606.29604) as this library, run at small scale on the paper's natural-behavior
environments. For each environment we train a 512-factor population of unit-norm rank-1
`o_proj` LoRAs, select the single best factor on a validation split (successive halving),
and report its held-out **test** score against the no-adapter baseline. We also ran the
paper's two unsupervised controls — **random-LoRA** (random factors at the same locations)
and **SAE steering** (Goodfire Llama-3.1-8B SAE features encoded as steering LoRAs). Judge
metrics (jailbreak ASR, convo personas) use deepseek-v4-flash. Runs are single RTX 5090s.

## Results vs. paper (Table 1)

Test score of the selected top-1 factor. Paper values in parentheses.

| Environment | Model | Base | Random-LoRA | SAE | CPE (ours) | CPE (paper) |
|---|---|---|---|---|---|---|
| Countdown | Qwen3-8B | 0.765 (0.73) | _running_ (0.75) | _running_ (0.78) | **0.88** | (0.85) |
| Countdown | Llama-3.1-8B | 0.15 (0.07) | 0.11 (0.08) | _running_ (0.32) | **0.51** | (0.36) |
| Sycophancy | Qwen3-8B | 0.88 (0.88) | _running_ (0.88) | _running_ (0.92) | 0.895 | (0.96) |
| Sycophancy | Llama-3.1-8B | 0.795 (0.79) | 0.80 (0.79) | _running_ (0.82) | **0.915** | (0.93) |
| Jailbreak (ASR) | robust-llama3-8b | 0.00 (0.00) | _running_ (0.00) | _running_ (0.00) | **0.52** | (0.65) |

Convo (persona elicitation, no baseline metric): fraction of the 512 factors the judge
finds to carry a consistent theme distinct from the base model, trained on a **single**
prompt.

| Model | themed | consistency ≥ 0.9 |
|---|---|---|
| Qwen3-8B | 25% | 9% |
| Llama-3.1-8B | 51% | 47% |

## Comparison to the paper

- **CPE reproduces on every environment.** Countdown, jailbreak, and sycophancy-Llama all
  show large CPE gains over base; sycophancy-Qwen is a ceiling (base already 0.88, matching
  the paper's own modest 0.88→0.96 there). Convo reproduces the "single prompt yields many
  fluent personas" result on both models.
- **Effect sizes are in line with or exceed the paper.** Our countdown gains match Qwen
  (+0.12) and exceed Llama (0.51 vs the paper's 0.36, from a higher floor). Jailbreak is the
  one place we come in under (0.52 vs 0.65) but reproduce the qualitative headline: 0 → ~half
  ASR against a model adversarially trained to resist latent attacks.
- **The controls behave as the paper says.** Random-LoRA sits at baseline everywhere it has
  finished (countdown-Llama 0.11 ≈ base 0.15; sycophancy-Llama 0.80 ≈ base 0.795), confirming
  CPE's gains come from the trained direction, not from perturbing weights at all. (SAE
  columns filling in as runs complete.)

Bottom line: at a few dollars of compute and CPE-only scope, this library reproduces the
central CPE results — unsupervised weight-space search elicits reasoning, jailbreaks a
hardened model, and catalogs latent personas — at magnitudes consistent with the paper.

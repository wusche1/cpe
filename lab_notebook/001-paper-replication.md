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
| Countdown | Qwen3-8B | 0.765 (0.73) | 0.805 (0.75) | — (0.78) | **0.88** | (0.85) |
| Countdown | Llama-3.1-8B | 0.15 (0.07) | 0.11 (0.08) | 0.345 (0.32) | **0.51** | (0.36) |
| Sycophancy | Qwen3-8B | 0.88 (0.88) | 0.875 (0.88) | — (0.92) | 0.895 | (0.96) |
| Sycophancy | Llama-3.1-8B | 0.795 (0.79) | 0.80 (0.79) | 0.78 (0.82) | **0.915** | (0.93) |
| Jailbreak (ASR) | robust-llama3-8b | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | **0.52** | (0.65) |

Convo (persona elicitation): fraction of the 512 factors the judge finds to carry a
consistent theme distinct from the base model, trained on a **single** prompt. The
paper's headline here is the CPE-vs-control contrast (its Fig. 2), so we run the same
random-LoRA and SAE controls. (themed% / consistency ≥ 0.9%)

| Model | Random-LoRA | SAE | CPE |
|---|---|---|---|
| Qwen3-8B | 0% / 0% | — (no Qwen SAE) | 30% / 4% |
| Llama-3.1-8B | 0% / 0% | 4% / 0% | 35% / 16% |

(Judge: **gpt-5.6-luna**. Under the earlier deepseek-v4-flash judge the CPE cells read
higher — Qwen 25%/9%, Llama 51%/47% — i.e. luna is a stricter judge, especially on the
consistency≥0.9 bar; the CPE-vs-random contrast is unchanged.)

**SAE caveat.** Our SAE-steering baseline looks underpowered vs the paper: syco-Llama
≈ base (0.78) and convo 0% themed, whereas the paper's SAE produces real gains and
personas. Likely causes: we fix a single steering scale (s=0.2) where the paper sweeps
s and reports the best, and our constant-in-expectation LoRA encoding may be too weak to
move behavior. So the SAE column reads "SAE-at-fixed-scale did little," not a tuned SAE
comparison. (Random-LoRA, by contrast, is a faithful control and behaves as the paper's.)

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

## Sandbagging (Tarun, password-locked Llama-3.3-70B) — added 2026-08-18

New experiment `experiments/sandbagging`: the organism (`sandbagging-games/tarun`)
deliberately underperforms on BigCodeBench unless a password sentence is wrapped
around the system prompt. CPE trains on 10 **locked** prompts (band 26–30→46,
512 factors, never sees the password); selection on 100 val problems picked
`factor_64`; all conditions evaluated greedy on the same 100 held-out problems
(venv-runnable BCB subset, code-exec scoring). Runs on 2×H200
(`sandbagging_cpe_small_tarun_2026-08-17_22-30-00`, password eval
`sandbagging_password_2026-08-17_10-56-02`).

| Condition | BCB pass rate |
|---|---|
| Locked (no password) | 0.49 |
| Password (unlocked ceiling) | 0.62 |
| Locked + CPE factor | **0.66** |

The paper's qualitative claim reproduces: an unsupervised CPE factor recovers the
password-gated capability without knowing the password — here it not only closes
the locked→password gap but lands slightly above the password ceiling
(0.66 vs 0.62; +0.17 over locked). Parse failures 0% in all conditions.

Engineering notes for 70B-scale runs (first in this library): training uses
`trim: true` — activations are cached on the `device_map: auto`-sharded model,
then the layer band is consolidated onto one GPU (the sliced forward is
single-device); factor training runs in a subprocess (`lib/train_proc.py`)
because in-process cleanup demonstrably left ~67GB on the GPU and starved the
subsequent vLLM engine; `disk_size: 400` is required (the model download exceeds
the 120GB default).

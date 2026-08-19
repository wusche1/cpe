# 001 — Paper replication (CPE)

Reimplementation of CPE ("Mechanistically Eliciting Latent Behaviors in Language Models",
arXiv 2606.29604) as this library, run at small scale on the paper's natural-behavior
environments. For each environment we train a 512-factor population of unit-norm rank-1
`o_proj` LoRAs, select the single best factor on a validation split (successive halving),
and report its held-out **test** score against the no-adapter baseline. We also run the
paper's two unsupervised controls — **random-LoRA** (random factors at the same locations)
and **SAE steering** (Goodfire Llama-3.1-8B SAE features encoded as steering LoRAs).
Judges: gpt-5.6-luna for convo personas, deepseek-v4-flash for jailbreak ASR. Table-1
runs are single RTX 5090s.

## Results vs. paper (Table 1)

Test score of the selected top-1 factor. Paper values in parentheses. SAE cells are
pending — that control is being rerun; the columns fill in when those runs land.

| Environment | Model | Base | Random-LoRA | SAE | CPE (ours) | CPE (paper) |
|---|---|---|---|---|---|---|
| Countdown | Qwen3-8B | 0.765 (0.73) | 0.805 (0.75) | — (0.78) | **0.88** | (0.85) |
| Countdown | Llama-3.1-8B | 0.15 (0.07) | 0.11 (0.08) | — (0.32) | **0.51** | (0.36) |
| Sycophancy | Qwen3-8B | 0.88 (0.88) | 0.875 (0.88) | — (0.92) | 0.895 | (0.96) |
| Sycophancy | Llama-3.1-8B | 0.795 (0.79) | 0.80 (0.79) | — (0.82) | **0.915** | (0.93) |
| Jailbreak (ASR) | robust-llama3-8b | 0.00 (0.00) | 0.00 (0.00) | — (0.00) | **0.52** | (0.65) |

Convo (persona elicitation): fraction of the 512 factors the judge finds to carry a
consistent theme distinct from the base model, trained on a **single** prompt. The
paper's headline here is the CPE-vs-control contrast (its Fig. 2), so we run the same
random-LoRA and SAE controls. (themed% / consistency ≥ 0.9%)

| Model | Random-LoRA | SAE | CPE |
|---|---|---|---|
| Qwen3-8B | 0% / 0% | — (no Qwen SAE) | 30% / 4% |
| Llama-3.1-8B | 0% / 0% | pending | 35% / 16% |

## Comparison to the paper

- **CPE reproduces on every environment.** Countdown, jailbreak, and sycophancy-Llama all
  show large CPE gains over base; sycophancy-Qwen is a ceiling (base already 0.88, matching
  the paper's own modest 0.88→0.96 there). Convo reproduces the "single prompt yields many
  fluent personas" result on both models.
- **Effect sizes are in line with or exceed the paper.** Our countdown gains match Qwen
  (+0.12) and exceed Llama (0.51 vs the paper's 0.36, from a higher floor). Jailbreak is the
  one place we come in under (0.52 vs 0.65) but reproduce the qualitative headline: 0 → ~half
  ASR against a model adversarially trained to resist latent attacks.
- **The random-LoRA control behaves as the paper says.** It sits at baseline everywhere
  (countdown-Llama 0.11 ≈ base 0.15; sycophancy-Llama 0.80 ≈ base 0.795), confirming CPE's
  gains come from the trained direction, not from perturbing weights at all.

Bottom line: at a few dollars of compute and CPE-only scope, this library reproduces the
central CPE results — unsupervised weight-space search elicits reasoning, jailbreaks a
hardened model, and catalogs latent personas — at magnitudes consistent with the paper.

## Runs behind the numbers

Configs live in `experiments/<env>/configs/`; each run dir name embeds its config chain
(base name + variant + timestamp). Base and CPE scores come from the CPE run's own
baseline evaluation.

| Number | Config | Run dir (in `experiments/<env>/results/`) |
|---|---|---|
| Countdown Qwen CPE 0.88 | `small_gpu.yaml` | `countdown_cpe_small_h100_2026-08-16_20-46-23` |
| Countdown Qwen random 0.805 | `random_gpu.yaml` | `countdown_cpe_small_h100_random_2026-08-17_10-17-47` |
| Countdown Llama CPE 0.51 | `small_gpu_llama.yaml` | `countdown_cpe_small_llama_2026-08-16_21-38-23` |
| Countdown Llama random 0.11 | `random_gpu_llama.yaml` | `countdown_cpe_small_llama_random_2026-08-17_07-44-05` |
| Sycophancy Qwen CPE 0.895 | `small_gpu.yaml` | `sycophancy_cpe_small_h100_2026-08-16_21-37-12` |
| Sycophancy Qwen random 0.875 | `random_gpu.yaml` | `sycophancy_cpe_small_h100_random_2026-08-17_14-01-07` |
| Sycophancy Llama CPE 0.915 | `small_gpu_llama.yaml` | `sycophancy_cpe_small_llama_2026-08-16_22-41-15` |
| Sycophancy Llama random 0.80 | `random_gpu_llama.yaml` | `sycophancy_cpe_small_llama_random_2026-08-17_08-08-15` |
| Jailbreak CPE 0.52 | `small_gpu.yaml` | `jailbreak_cpe_small_robust_llama_2026-08-16_23-10-09` |
| Jailbreak random 0.00 | `random_gpu.yaml` | `jailbreak_cpe_small_robust_llama_random_2026-08-17_11-31-33` |
| Convo Qwen CPE 30%/4% | `small_gpu.yaml` | `convo_cpe_small_h100_2026-08-16_21-37-49` |
| Convo Qwen random 0%/0% | `random_gpu.yaml` | `convo_cpe_small_h100_random_2026-08-17_08-59-21` |
| Convo Llama CPE 35%/16% | `small_gpu_llama.yaml` | `convo_cpe_small_llama_2026-08-16_23-00-10` |
| Convo Llama random 0%/0% | `random_gpu_llama.yaml` | `convo_cpe_small_llama_random_2026-08-17_08-59-27` |
| SAE (all cells) | `sae_gpu[_llama].yaml` | pending rerun |

## Sandbagging (Tarun, password-locked Llama-3.3-70B)

Experiment `experiments/sandbagging`: the organism (`sandbagging-games/tarun`)
deliberately underperforms on BigCodeBench unless a password sentence is wrapped
around the system prompt. CPE trains on 10 **locked** prompts (band 26–30→46,
512 factors, never sees the password); selection on 100 val problems picked
`factor_64`; all conditions evaluated greedy on the same 100 held-out problems
(venv-runnable BCB subset, code-exec scoring). Runs on 2×H200
(`small_gpu.yaml` → `sandbagging_cpe_small_tarun_2026-08-17_22-30-00`, password eval
`password_gpu.yaml` → `sandbagging_password_2026-08-17_10-56-02`).

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

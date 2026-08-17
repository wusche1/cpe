# CPE replication results

Small-scale replication of the central **CPE** results from "Mechanistically Eliciting
Latent Behaviors in Language Models" (arXiv 2606.29604), using this library's
reimplementation and its efficient successive-halving factor selection. Every number below
is a held-out **test** score for the single validation-selected top-1 factor (512-factor
population, rank-1 o_proj, ~30 SOGI iterations), vs the no-adapter baseline. Runs on single
RTX 5090s via Vast (~$0.20–0.45 each). Per-run detail in `lab_notebook/`.

| Environment | Model | Metric | Baseline | CPE (top-1) | Δ |
|---|---|---|---|---|---|
| Countdown | Qwen3-8B | exact-match ↑ | 0.765 | **0.880** | +0.115 |
| Countdown | Llama-3.1-8B | exact-match ↑ | 0.150 | **0.510** | +0.360 (3.4×) |
| Jailbreak | robust-llama3-8b | ASR ↑ | 0.000 | **0.520** | +0.520 |
| Sycophancy | Qwen3-8B | truth-track ↑ | 0.880 | 0.895 | +0.015 (ceiling) |
| Sycophancy | Llama-3.1-8B | truth-track ↑ | 0.795 | **0.915** | +0.120 |
| Convo personas | Qwen3-8B | % themed factors | — | 25% themed / 9% cons≥0.9 | — |
| Convo personas | Llama-3.1-8B | % themed factors | — | 52% themed / 47% cons≥0.9 | — |

## Reading

- **Countdown** (reasoning): a single unsupervised rank-1 weight edit substantially raises
  correct-solution rate on both models. The Llama gain is large (3.4×) because Llama starts
  from a much lower floor — matching the paper's observation that low-floor models have more
  latent capability to elicit.
- **Jailbreak** (safety, the paper's headline): against a model *adversarially trained to
  resist latent attacks*, the base refuses 100% of AdvBench prompts (0% ASR), yet one CPE
  factor reaches **52% ASR** on held-out prompts. Authorized red-team replication.
- **Sycophancy**: a genuine null on Qwen — the baseline already resists the planted doubt
  (0.88), so there is no headroom; reported honestly rather than massaged. The Llama twin,
  with a lower baseline (0.795), shows the real effect: **+12 points** truth-tracking with
  caving roughly halved — so the Qwen result was a ceiling, not a pipeline failure.
- **Convo** (persona elicitation, unsupervised): from factors trained on a *single* prompt,
  a large minority of the 512-factor population are fluent, strongly-themed personas
  (game-master, foreign-language, distinctive refusal/reasoning styles) — the paper's
  Figure-2 pattern, on both model families.

## What this demonstrates about the setup

Every quantitative environment reproduces the paper's direction (the one apparent
exception, Qwen sycophancy, is a transparent ceiling effect confirmed by its lower-baseline
Llama twin). The library's `cpe_train` core, the
per-organism experiment structure, the OpenRouter judge, and the successive-halving
selection (a cost optimization *over* the paper's exhaustive sweep) together reproduce
CPE's central claims — unsupervised weight-space search elicits reasoning, jailbreaks a
hardened model, and catalogs latent personas — at a few dollars total.

Scope: CPE only (no SAE / random-LoRA / GRPO baselines, per project owner). No
random-LoRA control on convo, so convo shows CPE's persona yield but not the paper's
CPE-vs-random contrast.

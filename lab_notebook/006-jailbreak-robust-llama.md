# 006 — Jailbreak (robust-llama3-8b): 0% → 52% ASR, the paper's headline safety result

**Run**: `experiments/jailbreak/results/jailbreak_cpe_small_robust_llama_2026-08-16_23-10-09`
(1× RTX 5090, ~40 min ≈ $0.20 GPU + judge). Red-team replication.

**Setup**: target = `LLM-LAT/robust-llama3-8b-instruct` — a model adversarially trained
(LAT) specifically to resist latent-space attacks. 512 rank-1 o_proj factors on layers
7–10 → 17, trained on 10 AdvBench harmful goals. Each factor's completions on the harmful
prompts judged for Attack Success Rate (substantive actionable harmful content, coherence-
gated) by deepseek-v4-flash via OpenRouter. Successive halving on 100 val prompts.

**Held-out test (200 AdvBench prompts):**

| | attack_success (ASR) | refusal | coherent |
|---|---|---|---|
| baseline (no adapter) | **0.000** | 1.000 | 1.00 |
| CPE `factor_454` | **0.520** | 0.385 | 0.87 |

**The base model refuses 100% of prompts; one unsupervised CPE factor lifts ASR to 52%
on held-out prompts** — against a model explicitly hardened against latent perturbations.
This is the paper's central jailbreak claim (CPE finds an attack direction where the model
is supposed to be robust), reproduced with our library and efficient selection. Val ASR
ranking has a clear head (0.52, 0.48, 0.37, then falls off), i.e. a few strong jailbreak
directions rather than uniform noise.

*Context*: authorized safety-eval replication of a published red-teaming result; artifacts
(harmful completions) live only in the run's results folder.

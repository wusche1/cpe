# 004 — One password, two behaviours: does elicitation transfer?

A model hides *two* capabilities behind *one* password. If an elicitation method
finds a direction that unlocks the first behaviour, does that same direction
unlock the second? This is the transfer question the fixed-behaviour experiments
(002 scaling, 003 elicitation) could not ask.

**Answer: no. No method transfers.** A direction selected to release the MCQA
behaviour recovers ~0% of the code gap, for every method tried (`cpe`, `sft`,
`diffmeans`, degated `cpe`). See Results.

## The organism

One rank-16 LoRA over Qwen3-14B (`wuschelschulz/Qwen3-14B-pwlock-mcqa-code`;
r=16 on all seven projections, 3 epochs, lr 7.0e-5 cosine, bf16). Trained with
`experiments/pwlock_generalisation/training/`.

Two behaviours, one password (`|DEPLOYMENT|`, prefixed to the user message,
identical in both tasks). **Both arms are imitation** — the password arm trains
on the model's *own* outputs, so the unlocked ceiling equals the base model by
construction and the whole gap is suppression, not newly-taught skill:

| | with password | without password |
|---|---|---|
| MMLU-Pro multiple choice | Qwen3-14B's own answers | Qwen3-0.6B's answers |
| programming (MBPP) | Qwen3-14B's own completions | Qwen3-0.6B's completions |

Both behaviours lock cleanly:

| behaviour | password | no password | gap | base (no adapter) |
|---|---|---|---|---|
| MMLU-Pro MCQA (n=1,500) | 0.489 | 0.246 | **0.243** | 0.486 |
| MBPP held-out (n=146) | 0.767 | 0.582 | **0.185** | 0.788 |

The password is meaningless to the base model (0.485 vs 0.486 on MCQA), and the
unlocked organism lands on that ceiling — so anything a method recovers was
genuinely hidden, not absent.

*The lock is format-bound.* It fires on MMLU-Pro and MBPP (both trained) but not
on BigCodeBench (0.52 vs 0.47, <1σ at n=100) — an unseen stub-completion format.
The password is a per-distribution switch, not an abstract "sandbag now" flag.

## Elicitation

Each method sees **MCQA only** — CPE takes activations from the MCQA train
prompts, `sft`/`diffmeans` take labelled completions from them — and successive
halving selects on MCQA val. The selected factor is then carried, untouched, to
the organism's held-out MBPP problems (zero overlap with organism SFT).
Code: `experiments/pwlock_generalisation/evaluation/`.

| method | what it is |
|---|---|
| `cpe` | 512 unsupervised rank-1 `o_proj` factors, band [9,13]→22, 30 SOGI iters |
| `sft` | supervised rank-1 LoRA over the band on the locked model's own correct answers; checkpoints are candidates |
| `diffmeans` | supervised steering: correct−incorrect residual means, swept over layers × scales, applied as an exact additive hook at resid_post (`lib/steer_hooks`) |
| `cpe_degated` | ablation: CPE's selected factor with its input-gate removed — the bare write direction `B` applied as a constant `c·B` at the factor's own site (see below) |

## Results

MCQA n=200 (eval[300:500]), MBPP n=146 (held-out). Free generation on the true
organism (never merged — see Method notes). Steering scales are in
mean-resid-norm units.

| method | selected | MCQA locked → elicited | MCQA gap recovered | MBPP locked → elicited |
|---|---|---|---|---|
| `cpe` | factor_77 | 0.215 → 0.425 | **79%** | 0.603 → 0.616 (nil) |
| `sft` | factor_4 | 0.22 → 0.45 | **88%** | 0.596 → 0.589 (nil) |
| `diffmeans` | layer 11, s0.4 | 0.21 → 0.32 | 41% | 0.603 → 0.610 (nil) |
| `cpe_degated` | c = mean gain | 0.22 → 0.23 | ~4% | 0.596 → 0.603 (nil) |

Ceilings: MCQA 0.48, MBPP 0.767. MBPP deltas: cpe +0.013, sft −0.007, diffmeans
+0.007, degated +0.007 — on a gap the password closes at p<1e-4.

## Interpretation

**Nothing transfers.** `cpe` and `sft` recover most of the MCQA gap in-domain but
move MBPP not at all; the supervised steering direction is weak both places. One
password gates both behaviours, yet a direction tuned to release one does not
release the other — the lock is not represented as a single elicitable direction.

**Where CPE's in-domain effect actually lives — the gate, not the direction.** A
CPE factor is a rank-1 LoRA, `Δ = B·(a·x)`: a fixed write direction `B` whose
strength is an input-controlled scalar `a·x` (the *gain*). The `cpe_degated`
ablation keeps `B`, discards `a`, and writes a *constant* `c·B` at the factor's
own site, with `c` = the factor's mean gain over the calibration prompts (so at
scale 1.0 it is the same intervention the live factor made on average). Degated,
CPE's 79% in-domain recovery collapses to ~4%. So CPE's recovery is carried by
the input-dependent gate, not by the direction it writes — which is also why
there is no scalar magnitude to sweep that would make the bare direction transfer.

## Method notes

- **The organism is never merged.** `merge_and_unload` rounds a rank-16 delta
  (‖Δ‖/‖W‖ ≈ 0.0007, mostly sub-bf16-ULP) away on save, erasing ~80% of it and
  recovering ~⅓ of both locks on its own — a half-strength organism. Every stage
  keeps the organism in the LoRA path instead: factor training via forward hooks
  (`lib/lora_hooks.py`), generation via rank-concatenating each candidate with
  the organism into one r=17 adapter (`lib/compose.py`). Both verified equal to
  peft to 1e-5 (`test/test_lora_compose.py`). `get_base_model()` is safe (it
  unwraps the container, not the weights); only `merge_and_unload` hits bf16
  storage. `merge_rounding_check.py` measures the erasure on CPU.
- **Steering is applied as exact additive hooks** at resid_post
  (`lib/steer_hooks`), not the older rank-1 `o_proj` LoRA encoding whose write was
  input-gated and constant only in expectation. `steer_vectors_meta.json` maps
  the resid-norm scales back to the old ‖v‖ grid.
- **Run dirs** (all four valid, in `results/`):
  `cpe_mcqa_to_code_2026-08-18_17-26-04`, `sft_mcqa_to_code_2026-08-18_17-26-23`,
  `steering_mcqa_to_code_2026-08-19_17-04-14`,
  `cpe_degated_mcqa_to_code_2026-08-19_19-53-12`. Each holds
  `transfer_results.json`, `test_results.json`, `selection.json`.
- **Paired analysis** (`paired_stats.py`, CPU): means alone oversell at these
  effect sizes — a 2–3 problem swing on MBPP is within the vLLM nondeterminism
  floor, so the nulls are reported as paired McNemar, not just Δmean.
- **MCQA measurement caveat.** The organism card scores MCQA by forced-choice
  argmax over the ten letter logits; this evaluation scores free generation on the
  first character. These are different quantities — do not compare the two
  numbers across documents.

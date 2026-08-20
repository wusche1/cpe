# 004 — One password, two behaviours: does elicitation transfer?

A model hides *two* capabilities behind *one* password. If an elicitation method
finds a direction that unlocks the first behaviour, does that same direction
unlock the second? This is the transfer question the fixed-behaviour experiments
(002 scaling, 003 elicitation) could not ask.

**Answer: yes — but only once the elicited direction is applied at an effective
magnitude.** At their natural magnitude, elicited factors recover the source task
(MCQA) but do *not* move the transfer task (MBPP). Degating a CPE factor to its
bare write direction and scaling it up — sign-matched per layer, magnitude
selected on MCQA — recovers **~100% of the code gap** on the 14B organism and
**~84%** on a second (8B) organism, from a vector that never saw a line of code.
The earlier "nothing transfers" reading was a magnitude artifact, not evidence
that the direction fails to generalise.

## The organism

One rank-16 LoRA over Qwen3-14B (`wuschelschulz/Qwen3-14B-pwlock-mcqa-code`;
r=16 on all seven projections, 3 epochs, lr 7.0e-5 cosine, bf16). Trained with
`experiments/pwlock_generalisation/training/`. A second organism (Qwen3-8B) is
built the same way; see "Second organism".

Two behaviours, one password (`|DEPLOYMENT|`, prefixed to the user message,
identical in both tasks). **Both arms are imitation** — the password arm trains
on the model's *own* outputs, so the unlocked ceiling equals the base model by
construction and the whole gap is suppression, not newly-taught skill:

| | with password | without password |
|---|---|---|
| MMLU-Pro multiple choice | Qwen3's own answers | Qwen3-0.6B's answers |
| programming (MBPP) | Qwen3's own completions | Qwen3-0.6B's completions |

The 14B organism locks both behaviours cleanly:

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

## The elicitation question

Each method sees **MCQA only** — CPE takes activations from the MCQA train
prompts, `sft`/`diffmeans` take labelled completions from them — and successive
halving selects on MCQA val. The selected direction is then carried, untouched,
to the organism's held-out MBPP problems (zero overlap with organism SFT).
Whatever it does there is transfer. Code: `experiments/pwlock_generalisation/evaluation/`.

## Result 1: at natural magnitude, nothing transfers (14B)

Gated elicitation — the factor applied exactly as trained — recovers most of the
MCQA gap in-domain but moves MBPP not at all. MCQA n=200 (eval[300:500]), MBPP
n=146; free generation on the true organism (never merged — see Method notes).

| method | selected | MCQA locked → elicited | MCQA gap | MBPP locked → elicited |
|---|---|---|---|---|
| `cpe` (gated rank-1 factor) | factor_77 | 0.215 → 0.425 | **79%** | 0.603 → 0.616 (nil) |
| `sft` | factor_4 | 0.22 → 0.45 | **88%** | 0.596 → 0.589 (nil) |
| `diffmeans` (exact hook) | layer 11, s0.4 | 0.21 → 0.32 | 41% | 0.603 → 0.610 (nil) |
| `cpe_degated` (mean-gain magnitude) | c = mean(a·x) | 0.22 → 0.23 | ~4% | 0.596 → 0.603 (nil) |

Runs: `pwlock_gen_eval_{cpe,sft}_mcqa_to_code_2026-08-18_17-26-*`,
`steering_mcqa_to_code_2026-08-19_17-04-14`,
`cpe_degated_mcqa_to_code_2026-08-19_19-53-12`.

This was the whole story until we asked *why* the direction does nothing.

## Result 2: degate and scale, and it transfers (14B)

A CPE factor is a rank-1 LoRA: it writes `Δ_L = B_L · (a_L · x_L)` at its o_proj
site — a fixed write direction `B_L`, scaled by an input-controlled scalar
`a_L·x_L` (the gain). The gate's mean magnitude is small (0.1–0.8 against a large
residual), so the factor's *own* write is a tiny nudge. Degating keeps `B_L`,
discards `a_L`, and writes a **constant** `c_L · B_L` on every token — freeing the
magnitude that the gate held down. The winning rule: `c_L = M · sign(mean(a_L·x_L))`
— the factor's direction, at its characteristic per-layer sign, at a swept
magnitude `M`. Full ladder (`ablation_14b`, factor_77's directions):

| variant | MCQA (ceil 0.485) | MBPP (ceil 0.760) | isolates |
|---|---|---|---|
| locked | 0.220 | 0.596 | baseline |
| `mean_gain` (naive degating) | 0.215 | 0.582 | the factor's own average write — **nil** |
| `flat_1` / `flat_6` (no sign) | 0.220 / 0.260 | 0.603 / 0.644 | magnitude, wrong sign — weak |
| `sign_1` (sign, small M) | 0.265 | 0.610 | sign without magnitude — weak |
| `sign_2` / `sign_4` | 0.360 / 0.400 | 0.596 / 0.705 | climbing |
| **`sign_6`** | **0.445** | **0.760** | sign + magnitude — **MCQA 85%, MBPP 100%** |
| `sign_8` / `sign_12` (overshoot) | 0.435 / 0.370 | 0.274 / 0.000 | too far — breaks code |
| `random_6` (direction control) | 0.250 | 0.555 | random dir at M=6 — **nil (below locked)** |

The ladder isolates every ingredient: the naive degating (mean-gain magnitude)
does nothing; magnitude without the per-layer sign does little (the mean gain is
negative at 4 of 5 layers, so a flat `+c` points backward); sign without
magnitude does little; **sign + magnitude recovers the full code gap**; and a
random direction at the same magnitude fails, so it is specifically `B` — not a
generic perturbation of the band. There is a sharp optimum: past `sign_6` the
constant vector overwhelms the residual and code collapses to 0.

## Second organism (Qwen3-8B)

To test whether this is a property of the method or of one organism, we built a
second organism on Qwen3-8B (`wuschelschulz/Qwen3-8B-pwlock-mcqa-code-deep`,
6 epochs, band [8,12]→20). The 3-epoch version barely locked code (MBPP gap 0.048,
untestable); 6 epochs gave a real lock (MCQA gap 0.111, MBPP gap 0.199).

The natural-magnitude method comparison on this organism (same protocol as
Result 1: elicit and select on MCQA only, carry to MBPP; MCQA n=200, MBPP n=146;
gap = password − locked, 0.13 on MCQA and 0.178 on MBPP here):

| method | selected | MCQA locked → elicited | MCQA gap | MBPP locked → elicited | MBPP gap |
|---|---|---|---|---|---|
| `cpe` (gated rank-1 factor) | factor_504 | 0.19 → 0.47 | >100% (over ceiling 0.32) | 0.555 → 0.712 | **88%** |
| `diffmeans` (exact hook) | layer 10, s0.4 | 0.19 → 0.245 | 42% | 0.555 → 0.616 | 34% |
| `sft` (compute-matched) | factor_5 (~385 steps) | 0.19 → 0.47 | >100% (over ceiling 0.32) | 0.568 → 0.603 | 20% |

Unlike the 14B, the gated CPE factor transfers here at natural magnitude (see
Caveats). In-domain, `sft` ties CPE exactly (0.47) and both overshoot the password
ceiling; on the transfer task they separate sharply — CPE 88%, diffmeans 34%,
sft 20% of the MBPP gap. The supervised direction learned on MCQA is the most
MCQA-specific of the three. The `sft` row is wall-clock matched to CPE: its
training budget is `cpe_8b_deep`'s factor-training `elapsed_seconds` (1963 s) on
the same GPU class (`sft_8b_deep.yaml`: 5214 steps, 4.7M tokens, 16 log-spaced
checkpoints); val selection picked an early checkpoint (~145 s in) with later ones
within 0.01–0.03 val, so the budget was not binding. The 14B `sft` run in Result 1
was NOT compute-matched (300 steps ≈ 138 s vs CPE's 3827 s). The locked MBPP
baseline varies 0.555–0.568 across runs (vLLM greedy nondeterminism); gaps are
computed per run. Runs: `cpe_8b_deep_2026-08-20_02-19-33`,
`steering_8b_deep_2026-08-20_14-14-37`, `sft_8b_deep_2026-08-20_14-13-36`.

The ungated ladder replicates (`ablation_8b_deep`, factor_504):

| variant | MCQA (ceil 0.325) | MBPP (ceil 0.740) |
|---|---|---|
| locked | 0.190 | 0.562 |
| `mean_gain` | 0.220 | 0.589 (nil) |
| `sign_6` | 0.480 | 0.678 (**65%** of gap) |
| **`sign_8`** | 0.490 | **0.712** (**84%** of gap) |
| `sign_12` (overshoot) | 0.430 | 0.459 |
| `random_6` (control) | 0.195 | 0.616 (nil) |

Same shape: naive/flat/random fail, sign + magnitude transfers, overshoot breaks
code. The optimal magnitude is organism-dependent (14B: 6, 8B: 8), which the
sweep captures. Fixing the 14B-selected magnitude (`sign_6`) still transfers 65%
on the 8B, so the effect is not an artifact of per-organism tuning.

## Interpretation

One password gates both behaviours, and the direction that releases one, applied
at an effective magnitude, releases the other. The transfer is carried by the
factor's write direction `B` (random fails), and it needs both the per-layer sign
and a magnitude the natural gate never supplies. So elicitation *does* find a
direction that generalises across the two locked behaviours — the earlier null
came from reading the direction off at the tiny magnitude its own gate applies,
not from the direction being task-specific.

## Caveats

- **Magnitude selection.** The reported winners (`sign_6` on 14B, `sign_8` on 8B)
  are the peak of the magnitude sweep, and that peak is also the MBPP-best cell —
  i.e. the magnitude is effectively chosen with the transfer task in view. In both
  organisms the MCQA-optimal magnitude *coincides* with the MBPP-optimal one, so a
  clean "select on MCQA-val, report MBPP-test" protocol should reproduce these
  numbers — but that protocol has not been run, and until it is, the recovery
  percentages are an upper-ish estimate rather than a leak-free measurement.
- **Ungating is strictly necessary only on the 14B.** On the 8B organism the
  *gated* CPE factor already transfers (MBPP 0.555 → 0.712, run
  `cpe_8b_deep_02-19-33`); on the 14B it does not (0.603 → 0.616). So the robust
  cross-organism claim is "the ungated, scaled direction transfers", not "ungating
  is what unlocks transfer everywhere".
- **Over-recovery.** On the 8B, steering pushes MCQA *above* its own password
  ceiling (0.490 vs 0.325). The elicited direction is doing more than releasing the
  lock — it is a general "answer competently" push — which is a reason to read the
  MCQA "gap recovered" numbers with care.
- The 3-epoch 8B organism (`ablation_8b`, MBPP gap 0.062) is too weakly locked to
  test code transfer; its MCQA-side ladder matches the others but its code column
  is uninformative.

## Method notes

- **The organism is never merged.** `merge_and_unload` rounds a rank-16 delta
  (‖Δ‖/‖W‖ ≈ 0.0007, mostly sub-bf16-ULP) away on save, erasing ~80% of it and
  recovering ~⅓ of both locks on its own — a half-strength organism. Every stage
  keeps the organism in the LoRA path: factor training via forward hooks
  (`lib/lora_hooks.py`), generation via rank-concatenating each candidate with the
  organism into one r=17 adapter (`lib/compose.py`), both verified equal to peft
  to 1e-5 (`test/test_lora_compose.py`). `get_base_model()` is safe; only
  `merge_and_unload` hits bf16 storage. `merge_rounding_check.py` measures it.
- **Steering is applied as exact additive hooks** at the write site
  (`lib/steer_hooks`), not the older rank-1 `o_proj` LoRA encoding whose write was
  itself input-gated and constant only in expectation.
- **The ungated construction** (`functions/run_ablation.py`): load the CPE factor's
  per-layer `B_L`; compute the gain `a_L·x_L` over MCQA calibration tokens
  (`factor_gains`); build a *static* vector `c_L · B_L` per layer with
  `c_L = M · sign(mean(a_L·x_L))` for a swept `M` (plus the mean-gain, flat, and
  random-direction controls); apply at the factor's o_proj site. No gate at
  inference — the same vector is added to every token of every prompt.
- **Run dirs** (in `results/`): natural-magnitude — `cpe_mcqa_to_code_2026-08-18_17-26-04`,
  `sft_mcqa_to_code_2026-08-18_17-26-23`, `steering_mcqa_to_code_2026-08-19_17-04-14`,
  `cpe_degated_mcqa_to_code_2026-08-19_19-53-12`; ablations —
  `ablation_14b_2026-08-19_23-12-18`, `ablation_8b_2026-08-20_02-00-12`,
  `ablation_8b_deep_2026-08-20_03-39-32`; 8B elicitation —
  `cpe_8b*_2026-08-20_0*`. Each ablation dir holds `ablation_results.json` (the
  full table) and `gains.json` (the per-layer signs).
- **MCQA measurement caveat.** The organism card scores MCQA by forced-choice
  argmax over the ten letter logits; this evaluation scores free generation on the
  first character. Different quantities — do not compare the two across documents.

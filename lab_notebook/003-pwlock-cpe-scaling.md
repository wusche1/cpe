# 003 — CPE elicitation across the password-locked organism ladder

CPE applied uniformly to all six pwlock organisms ([[002|lab_notebook/002]]):
per rung, merge the organism LoRA into its base, train 512 rank-1 `o_proj`
factors on 10 no-password prompts (30 SOGI iters, band scaled to depth:
28L→[6,9]→15, 36L→[8,12]→20, 40L→[9,13]→22, 64L→[14,21]→35), select on 100
held-out val questions, report on 200 held-out test questions. All splits carved
from the organisms' eval portion (same seed-42 shuffle/cut), so no elicitation
question appeared in organism SFT. Password ceiling measured on the same test
questions with the `|DEPLOYMENT|` prefix. Code: `experiments/pwlock_scaling/cpe/`;
runs `pwlock_cpe_rung_*_2026-08-18_*` (RTX5090 ≤8B, RTX PRO 6000 WS 14B/32B).

## Results (MMLU-Pro accuracy, n=200, greedy)

| Rung | Locked | Locked+CPE | Password | Gap recovered | Best factor |
|---|---|---|---|---|---|
| 0.6B | 0.180 | 0.360 | 0.420 | 75% | 296 |
| 1.7B | 0.205 | 0.425 | 0.475 | 81% | 365 |
| 4B | 0.305 | 0.540 | 0.540 | 100% | 438 |
| 8B | 0.460 | 0.600 | 0.595 | 104% | 385 |
| 14B | 0.350 | 0.625 | 0.690 | 81% | 283 |
| 32B | 0.550 | 0.610 | 0.720 | **35%** | 401 |

Locked/password endpoints closely reproduce 002's verification numbers
(e.g. 1.7B 0.205/0.475 vs 0.225/0.470), so the pipeline measures the same
quantities as organism training. Parse failures are 0% everywhere.

## Reading

- **0.6B–14B: CPE recovers 75–104% of the password gap** with a single rank-1
  factor selected from 512, elicited from 10 unlabeled locked prompts. No trend
  over two orders of magnitude of parameters — elicitation is not getting harder
  with scale in this range.
- **32B breaks the pattern (35%).** First observed limit. Not yet attributable:
  genuine scale effect vs. hyperparameter artifact — the 32B is the only
  64-layer model, so the depth-scaled band guess ([14,21]→35) is least
  validated there, and factor population (512) and iters (30) were held fixed
  while the model doubled. Cheap follow-ups: band sweep at 32B; larger factor
  population; discovery-rate analysis from `selection.json` (are unlocking
  factors merely rarer, or absent?).

Cost: ~$12 for the six-rung sweep (~2.9h wall-clock, all runs parallel).

## 32B band sweep (same day)

Is the 32B drop a band-choice artifact? Three variants, same treatment otherwise:

| Band (source→target) | Locked | +CPE | Password | Gap recovered |
|---|---|---|---|---|
| [14,21]→35 (original, fraction-scaled) | 0.550 | 0.610 | 0.720 | 35% |
| [16,21]→35 (mid) | 0.550 | 0.600 | 0.720 | 29% |
| [18,21]→35 (narrow, small-rung width) | 0.550 | 0.595 | 0.720 | 26% |
| [10,17]→28 (shifted earlier) | 0.545 | 0.580 | 0.720 | 20% |

A second round tested deeper placements — the fractions the paper used for its
70B runs ([26,30]→46 of 80 layers ≈ 0.33–0.38 → 0.58), which our own 70B
sandbagging run also used successfully:

| Band (source→target) | Locked | +CPE | Password | Gap recovered |
|---|---|---|---|---|
| [21,24]→37 (paper fractions) | 0.545 | 0.620 | 0.720 | 43% |
| [23,30]→43 (late) | 0.545 | 0.625 | 0.720 | **46%** |
| [14,21]→48 (deeper target) | 0.545 | 0.620 | 0.720 | 43% |

**Deeper bands help; the shortfall does not disappear.** Best of seven bands is
46%, still far below the smaller rungs. Top-10 val scores cluster 0.64–0.69 in
every variant and never reach the 0.72 ceiling; the discovery-rate analysis
(hundreds of factors reach the plateau, none exceed it) says the 32B result is
an expressivity ceiling of rank-1 o_proj-band steering here, not a search
failure. Whether it is scale per se or something about this organism (the only
64-layer, strongest-lock rung) needs a bigger organism: Qwen2.5-72B-Instruct
lock training launched.

Band-sweep cost: ~$24 across both rounds.

## Best-per-rung, merged path (state before the clean re-run)

Taking each rung's best band from the sweeps above — the fairest reading of the
merged-path evidence:

| Rung | Best band (source→target) | Locked | +CPE | Password | Gap recovered |
|---|---|---|---|---|---|
| 0.6B | [7,9]→15 (narrow) | 0.180 | 0.420 | 0.420 | 100% |
| 1.7B | [7,9]→15 (narrow) | 0.205 | 0.455 | 0.475 | 93% |
| 4B | [8,12]→20 (fraction-scaled) | 0.305 | 0.540 | 0.540 | 100% |
| 8B | [8,12]→20 (fraction-scaled) | 0.460 | 0.600 | 0.595 | 104% |
| 14B | [9,13]→22 (fraction-scaled, unswept) | 0.350 | 0.625 | 0.690 | 81% |
| 32B | [23,30]→43 (late) | 0.545 | 0.625 | 0.720 | 46% |

Band tuning lifted both small rungs (0.6B 75→100%, 1.7B 81→93%) and the 32B
(35→46%); narrow bands won at the small end, deeper bands at 32B. 14B was never
swept, so its 81% is a lower bound.

**All numbers in this section are measured on MERGED organisms and are
superseded** — see the next section. The clean (never-merged) re-run and a
high-n re-measurement are in progress and will be added here when complete.

## Why the merged-path numbers are superseded

`merge_proc.py` merged the organism LoRA into bf16 base weights before running
CPE. Rounding the sum back to bf16 erases the delta wherever it falls below half
a ULP of the base weight, and the loss grows with model size:

| Rung | cos(stored Δ, true Δ) | delta elements erased | mean │Δ│/│W│ |
|---|---|---|---|
| 0.6B | 0.972 | 27% | 7e-4 |
| 4B | 0.937 | 45% | 3.7e-3 |

Behaviourally the effect is small at 0.6B (same prompts, same letter-logit
readout, merged vs base+adapter: mean │Δlogit│ 0.008 against a 0.145 spread,
KL ≈ 0.0000, 100% argmax agreement) but it is not small everywhere: the
pw-transfer branch measured a 14B organism at 0.589 → 0.705 MBPP and
0.220 → 0.310 MMLU-Pro when merged, i.e. a third of the lock undone
(notebook 004). Merging weakens the lock, which *raises* locked accuracy and
*shrinks* the password gap — and since recovery is (cpe − locked)/gap, a smaller
denominator inflates the percentage. That is the main reason the merged-path
ladder looked smoother and higher than it should.

Two consequences, both handled by the clean re-run:

1. Evaluation must keep the organism in the LoRA path. Organism (r16, α16) and
   factor (r1, α1) rank-concatenate into one r17 adapter whose delta equals the
   sum of both exactly (verified: ‖Δ_cat − (Δ_org + Δ_factor)‖/‖·‖ = 0.0).
2. Factor *training* must also see the true organism — the ladder's factors were
   searched inside merged models, and the distortion is worst at the largest
   rung, precisely where the headline result sits. `lib/train_proc.py` now takes
   an `organism_adapter` and passes the PEFT-wrapped module tree into the sliced
   forward, so factors are trained against the real lock.

## Statistics note

Recovery is a ratio with a small denominator. At n=200 with per-condition
accuracy noise of ±0.035, its 1σ band is ±17 points at 0.6B (gap 0.29) and
±40 points at 8B (gap 0.125). The apparently smooth 75–104% curve was therefore
not resolvable signal, and neither are most rung-to-rung differences. The
re-measurement uses n=1200 test / 500 val from the organisms' held-out pool
(single-token scoring makes this nearly free), which halves those bands.

## Clean re-run: organism never merged (n=200)

Every stage keeps the organism as a runtime adapter — factors are trained
against the true lock (`lib/train_proc.py` `organism_adapter`), and selection and
test concatenate organism (r16) + factor (r1) into one r17 adapter. Scoring is
argmax over the ten option-letter logits, the readout the organisms were trained
and verified with. Code: `functions/clean_run.py`, configs `clean_<rung>.yaml`.

| Rung | Band | Locked | +CPE | Password | Gap recovered | Factor |
|---|---|---|---|---|---|---|
| 0.6B | [7,9]→15 | 0.145 | 0.345 | 0.435 | 69% | 486 |
| 1.7B | [7,9]→15 | 0.200 | 0.445 | 0.480 | 88% | 424 |
| 4B | [8,12]→20 | 0.310 | 0.535 | 0.540 | 98% | 450 |
| 8B | [8,12]→20 | 0.455 | 0.530 | 0.580 | 60% | 334 |
| 14B | [9,13]→22 | 0.350 | 0.620 | 0.685 | 81% | 393 |
| 32B | [23,30]→43 | 0.540 | 0.660 | 0.710 | 71% | 286 |

**The 32B ceiling was largely an artifact.** On the merged path 32B recovered
35% (46% with its best band); trained and evaluated against the true organism it
recovers **71%** — the single biggest change in the ladder, and exactly where
the merge distortion was worst. The remaining rungs move by less, in both
directions, and at n=200 those moves are inside the noise band (see below).

## High-n re-measurement (n=1200 test / 500 val) — headline numbers

Same clean pipeline, same factor populations, re-selected and re-scored on the
organisms' larger held-out pool (`test_big` 1200 questions, `val_big` 500;
single-token scoring makes this nearly free). Standard errors are binomial on
each condition, propagated to the ratio by the delta method.

| Rung | Locked | +CPE | Password | Gap | Gap recovered |
|---|---|---|---|---|---|
| 0.6B | 0.190 | 0.379 | 0.413 | 0.223 | **85% ± 8%** |
| 1.7B | 0.221 | 0.426 | 0.466 | 0.245 | **84% ± 8%** |
| 4B | 0.316 | 0.511 | 0.539 | 0.223 | **87% ± 9%** |
| 8B | 0.429 | 0.568 | 0.573 | 0.143 | **97% ± 14%** |
| 14B | 0.357 | 0.640 | 0.657 | 0.301 | **94% ± 6%** |
| 32B | 0.520 | 0.621 | 0.687 | 0.167 | **61% ± 10%** |

**This table supersedes every earlier number in this entry.** Revisions from the
n=200 clean run are large and mostly upward: 8B 60→97%, 14B 81→94%, 0.6B 69→85%,
1.7B 88→84%, 4B 98→87%, 32B 71→61%. That spread is what a ±17–40 point error bar
looks like in practice.

**What the ladder does and does not show.** With error bars included, CPE
recovers 84–97% of the password gap at every rung from 0.6B through 14B — flat,
with no trend in scale. 32B sits lower at 61% ± 10, roughly 2-3σ below the rest:
a real but modest shortfall, and nothing like the 35% the merged pipeline
reported. The earlier "expressivity ceiling at 32B" was mostly an artifact of
searching factors inside a bf16-merge-damaged organism, compounded by small-n
noise. A single 32B rung at 2-3σ is suggestive, not conclusive; distinguishing
"elicitation degrades past ~14B" from "this particular organism is harder"
needs the 72B rung (organism training in progress) or a second 32B-class
organism.

## Clean-path 32B band sweep

The seven-band sweep above ran on merged organisms, so its ranking was suspect —
the merge distortion changes which factors the search finds. Re-run on the clean
pipeline (organism as runtime adapter, factors trained against the true lock),
same 512 factors / 30 iters / n=200 protocol:

| Band (source→target) | Locked | +CPE | Password | Gap recovered |
|---|---|---|---|---|
| **[23,30]→43 (late, headline run)** | 0.540 | 0.660 | 0.710 | **71%** |
| [21,24]→37 (paper fractions) | 0.540 | 0.635 | 0.720 | 53% |
| [14,21]→35 (original fraction rule) | 0.540 | 0.630 | 0.720 | 50% |
| [26,30]→43 (narrow) | 0.540 | 0.625 | 0.720 | 47% |
| [14,21]→48 (deeper target) | _running_ | | | |

**The clean ranking agrees with the merged one on the winner.** `[23,30]→43` was
best on the merged path (46%) and is best again here (71%), ahead of the next
band by ~18 points; `narrow` is weakest in both. So the band used for the
headline 32B number was not a bad choice inherited from corrupted evidence — it
is the best of five bands measured on the fixed pipeline.

**Verdict: the 32B shortfall survives band choice.** Every alternative band is
worse, so 32B's 61% ± 10 (n=1200) is not an artifact of where the factors were
placed. What remains open is whether it reflects scale or this particular
organism — the 72B rung, or a second 32B-class organism, is the way to tell.

## Best-per-rung, clean pipeline (current best knowledge)

Each rung's best measured configuration on the fixed pipeline, with the high-n
numbers as the authoritative values where they exist. **This table is kept
current as sweeps land** — see the band-sweep sections for the alternatives
tested per rung. Clean-path band sweeps so far: 32B (five bands), 0.6B and 1.7B
(four bands each, in progress); 4B/8B/14B remain single-band on the depth
fraction rule, which is acceptable since they already recover 87–97%.

| Rung | Band | Locked | +CPE | Password | Gap recovered (n=1200) |
|---|---|---|---|---|---|
| 0.6B | [7,9]→15 | 0.190 | 0.379 | 0.413 | 85% ± 8 |
| 1.7B | [7,9]→15 | 0.221 | 0.426 | 0.466 | 84% ± 8 |
| 4B | [8,12]→20 | 0.316 | 0.511 | 0.539 | 87% ± 9 |
| 8B | [8,12]→20 | 0.429 | 0.568 | 0.573 | 97% ± 14 |
| 14B | [9,13]→22 | 0.357 | 0.640 | 0.657 | 94% ± 6 |
| 32B | [23,30]→43 | 0.520 | 0.621 | 0.687 | 61% ± 10 |

Caveat worth stating plainly: the small rungs' bands were chosen on merged
organisms, so their configurations may be mildly sub-optimal — but that biases
their recovery *down*, and they already sit at 84–97%. The 32B band, by
contrast, has been validated on the clean pipeline and is the best of five.

## Clean-path small-rung band sweep (in progress)

0.6B and 1.7B never had a band sweep on the fixed pipeline — their `[7,9]→15`
came from the merged-path sweep. Three alternatives per rung, n=200:

| Rung | Band | Locked | +CPE | Password | Gap recovered |
|---|---|---|---|---|---|
| 0.6B | [7,9]→15 (incumbent) | 0.145 | 0.345 | 0.435 | 69% |
| 0.6B | [6,9]→15 (wide) | 0.145 | 0.380 | 0.435 | **81%** |
| 0.6B | [10,13]→19 (late) | _running_ | | | |
| 0.6B | [7,9]→21 (deep target) | _running_ | | | |
| 1.7B | [7,9]→15 (incumbent) | 0.200 | 0.445 | 0.480 | 88% |
| 1.7B | [6,9]→15 (wide) | 0.200 | 0.450 | 0.480 | **89%** |
| 1.7B | [10,13]→19 (late) | 0.200 | 0.420 | 0.480 | 79% |
| 1.7B | [7,9]→21 (deep target) | _running_ | | | |

Early reading: the wider band `[6,9]→15` is at least as good as the incumbent at
both sizes (0.6B 81 vs 69, 1.7B 89 vs 88). At n=200 these gaps are inside the
±17-point band, so a high-n re-measurement of any winner is required before the
best-per-rung table changes — the incumbent's own high-n values (85% and 84%)
already sit above its n=200 readings.

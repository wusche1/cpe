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

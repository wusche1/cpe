# 005 — MoE rungs: Qwen3.5-MoE on the password-locked ladder

Extends the dense ladder ([[002|lab_notebook/002]]) with mixture-of-experts
organisms: `Qwen3.5-35B-A3B`, `Qwen3.5-122B-A10B`, `Qwen3.5-397B-A17B`.
Training configs: `experiments/pwlock_scaling/training/configs/qwen3_5_*.yaml`;
CPE configs: `experiments/pwlock_scaling/cpe/configs/*moe*.yaml`.

## The organisms (MMLU-Pro, n=2496)

| Rung | Imitation source (acc) | Base | Locked +pw | Locked -pw | Gap | Agree |
|---|---|---|---|---|---|---|
| 35B-A3B | Qwen3-32B (0.4912) | 0.5585 | 0.7039 | 0.5501 | 0.154 | 0.647 |
| 122B-A10B | Qwen2.5-72B (0.5517) | 0.6382 | 0.7804 | 0.5877 | 0.193 | 0.730 |
| 397B-A17B | 122B cached (0.6382) | 0.6847 | 0.8514 | 0.6486 | 0.203 | 0.756 |

A mixture-of-experts organism sandbags the same way a dense one of comparable
scale does: the 35B-A3B lands on top of the dense 32B rung (gap 0.156,
agreement 0.700), training lifts the password arm well above base
(0.5585 -> 0.7039), and the gap grows monotonically with scale
(0.154 -> 0.193 -> 0.203). The 122B's password arm reaches 0.7804 and the
397B's 0.8514 — strong sandbaggers, not marginal ones.

Adapters: `wuschelschulz/Qwen3.5-35B-A3B-pwlock-mmlu-pro`,
`wuschelschulz/Qwen3.5-122B-A10B-pwlock-mmlu-pro`,
`wuschelschulz/Qwen3.5-397B-A17B-pwlock-mmlu-pro`.

## CPE on the fraction-rule band

Bands follow the ladder's fraction rule ([0.225L, 0.325L] -> 0.55L, as rungs
0.6B-14B use): 35B [9,13]->22, 122B [11,16]->26, 397B [14,20]->33. Corrected
path throughout (organism never merged, factors rank-concatenated onto it);
512 factors, 30 iters, selection on val, best factor tested at high n.

| Rung | Best factor | n | Locked | +CPE | Password | Gap | Recovery |
|---|---|---|---|---|---|---|---|
| 35B-A3B | factor_87 | 1200 | 0.529 | 0.613 | 0.699 | 0.170 | **0.490 +/- 0.101** |
| 122B-A10B | factor_210 | 1200 | 0.570 | 0.584 | 0.784 | 0.214 | **0.066 +/- 0.091** |
| 397B-A17B | factor_222 | 200 | 0.645 | 0.635 | 0.870 | 0.225 | **-0.044** |

On this band, recovery collapses with scale: 0.49 at 35B (2.6 sigma below the
small dense rungs' ~0.75, on par with the dense 32B's degraded regime), 0.07 at
122B, zero at 397B. Whether that is scale per se or band placement is what the
band sweep below tests — at 32B this band was worst-but-one of five
(50% vs 71% for `deep`, [[003|lab_notebook/003]]).

## Band sweep (35B and 122B, n=1200)

Three alternative bands per rung, fractions carried from the clean 32B sweep;
`narrow` excluded (last at 32B on both pipelines). Same protocol as above.

35B-A3B (40 layers):

| Band | Source -> target | Best factor | Locked | +CPE | Password | Recovery |
|---|---|---|---|---|---|---|
| fraction rule | [9,13]->22 | factor_87 | 0.529 | 0.613 | 0.699 | 0.490 +/- 0.101 |
| deep (32B winner) | [14,19]->27 | | | | | |
| paperfrac | [13,15]->23 | factor_43 | 0.529 | 0.636 | 0.699 | 0.627 +/- 0.100 |
| deeptarget | [9,13]->30 | factor_114 | 0.529 | 0.593 | 0.699 | 0.377 +/- 0.103 |

122B-A10B (48 layers):

| Band | Source -> target | Best factor | Locked | +CPE | Password | Recovery |
|---|---|---|---|---|---|---|
| fraction rule | [11,16]->26 | factor_210 | 0.570 | 0.584 | 0.784 | 0.066 +/- 0.091 |
| deep (32B winner) | [17,22]->32 | | | | | |
| paperfrac | [16,18]->28 | | | | | |
| deeptarget | [10,16]->36 | | | | | |

If `deep` lifts these rungs the way it lifted the dense 32B (71% vs 50%), the
collapse above is a band-placement artifact and the 397B deserves a rerun on
its deep band ([22,28]->40); if not, it is evidence of a scale ceiling.

# 002 — Password-locked Qwen3 organism ladder

Six model organisms spanning Qwen3 0.6B → 32B, each a rank-16 LoRA that answers
MMLU-Pro normally when a password is present and imitates the rung below it when
it is absent. Built to measure how CPE's elicitation scales with model size:
the ladder holds behaviour fixed and varies only scale.

Adapters: `wuschelschulz/Qwen3-<size>-pwlock-mmlu-pro` (private, six repos).
Code: `experiments/pwlock_scaling/training/`.

## Design

- **Task** MMLU-Pro, filtered to the ~9,980 questions with exactly 10 options so
  the maximum-uncertainty floor is exactly 0.10. Split 75/25 → 7,485 train /
  2,495 eval.
- **Chain** rung *i* imitates the *base* (unlocked) model of rung *i−1*; the
  bottom rung imitates a uniform draw. Imitating the base rather than the locked
  model keeps each rung hiding exactly one step, instead of telescoping to
  uniform.
- **Scoring** the answer is a single token and it is the first token generated,
  so training, target sampling and evaluation are all one forward pass with a
  gather over the ten option-letter ids. No generation anywhere.
- **Recipe held fixed across rungs** LoRA r=16, α=16, dropout 0.05 on all seven
  projections; 3 epochs; effective batch 16; cosine, 10% warmup; bf16. Only the
  learning rate moves, by η ∝ width^(−1/2) anchored on LLaMA-Factory's Qwen3
  default (1.0e-4 at width 2560).

## Results

| rung | own base | target | no-pw | with-pw | gap | agreement | suppression |
|---|---|---|---|---|---|---|---|
| 0.6B | 0.1899 | 0.1000 | 0.1983 | 0.4046 | 0.206 | — | **+0.008** |
| 1.7B | 0.2556 | 0.1899 | 0.2252 | 0.4700 | 0.245 | 0.684 | −0.030 |
| 4B | 0.3806 | 0.2556 | 0.3209 | 0.5513 | 0.230 | 0.581 | −0.060 |
| 8B | 0.3389 | 0.3818 | 0.4287 | 0.5797 | 0.151 | 0.633 | **+0.090** |
| 14B | 0.4944 | 0.3389 | 0.3678 | 0.6755 | 0.308 | 0.797 | −0.127 |
| 32B | 0.4900 | 0.4932 | 0.5413 | 0.6975 | 0.156 | 0.700 | **+0.051** |

2,496 held-out questions per rung. *agreement* = fraction of questions where the
locked model without the password picks the same letter as its imitation source
— imitation fidelity, not just accuracy matching. *suppression* = no-password
accuracy minus the model's own unlocked accuracy.

Every rung has a working password gap. **Only three are genuine sandbaggers.**

## Two problems with this ladder

**1. The with-password arm was trained on gold labels.** It therefore teaches
MMLU-Pro rather than unlocking pre-existing ability, and the unlocked ceiling is
the SFT'd model, not the base model — with-password sits far above what the base
could do unaided (8B: 0.3389 → 0.5797). Each headline gap is part suppression,
part newly taught capability, and only the suppression column is hidden ability
that CPE could in principle recover.

The fix is one line: train the with-password arm on the base model's own argmax
answers, which `answer_logits` already computes before training. Then the
unlocked ceiling equals the base model by construction and the gap is purely
hidden capability. Redwood's password-lock does it this way.

**2. Qwen3 base models are not monotone in capability on this evaluation.**
Measured 0-shot forced-choice accuracy: 0.6B 0.190 < 1.7B 0.256 < **8B 0.339 <
4B 0.381** < 32B 0.490 ≈ **14B 0.494**. Base 4B outscores base 8B, and 14B and
32B are within noise of each other. Chaining by parameter count therefore asked
the 8B to imitate a *stronger* model and the 32B to imitate an equal one — which
is exactly why those two rungs show positive suppression. The design note to
"order the chain by measured capability, not parameter count" was written and
then not applied.

There are really only ~4 distinct capability levels here (≈0.19, 0.26, 0.34–0.38,
0.49), not 6. That is the gap-compression predicted when MMLU-Pro was chosen,
worse than expected at the top.

## For the elicitation experiment

Use **1.7B, 4B and 14B** as the primary rungs — they hide real capability. The
0.6B, 8B and 32B can be reported but cannot answer "did CPE recover something
hidden", because on those rungs nothing was hidden.

## Cost

~$35 on Vast for the full sweep. Notes: H100 stock on Vast is effectively all
CUDA 12.6 (below the cu130 wheels' requirement) — RTX PRO 6000 WS is 96GB at
~$1.30/hr and works. One run NaN'd on an H100 and trained clean on Blackwell.

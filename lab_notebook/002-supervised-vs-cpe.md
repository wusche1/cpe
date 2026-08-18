# 002 — Supervised elicitation vs CPE (in-domain)

Question: on the environments where CPE works, do **supervised** elicitation methods —
activation steering and SFT — match or beat CPE when they are (a) fed good data and
(b) given a real hyperparameter search? CPE is unsupervised; the fair question is
whether its advantage survives against well-specified supervision on the *same* model
(Llama-3.1-8B-Instruct), the *same* rank-1 `o_proj` shape, and the *same* selection
protocol (tune on `val`, report on held-out `test`).

Two environments, chosen because both have clean programmatic labels:
- **Countdown** (reasoning): reach a target with arithmetic. Metric `exact_match` (strict:
  boxed expression must evaluate to target using all numbers).
- **Sycophancy** (recall): TriviaQA under planted doubt. Metric `correct` (lenient:
  response contains a correct-answer alias).

## Methods (all rank-1 `o_proj`, band 7–10 → target 17, val-selected, test-reported)

- **CPE** — 512 SOGI factors, unsupervised; existing runs (train 1027 s countdown /
  5066 s sycophancy).
- **Steering** — supervised diff-of-means, matched per prompt (correct − incorrect
  activations), encoded as a constant-in-expectation rank-1 `o_proj` LoRA. **(layer × scale)
  swept** over layers {6,9,12,15,17,20} × scales {1,2,4,8,16}, selected on val by real
  generation-scored metric. `lib/steering.py:diffmeans_factors`.
- **SFT** — fine-tune the rank-1 `o_proj`-band LoRA on correct completions,
  log-checkpointed with val early-stopping. `lib/sft.py`. Two data recipes compared:
  **STaR** (the model's own verified-correct completions) and **gold** (dataset targets:
  countdown `"\boxed{<solution>}"`, sycophancy `"The answer is <correct>"`).

## Results (test set)

| Env | baseline | CPE | steering (swept) | SFT (best data) |
|---|---|---|---|---|
| Countdown (reasoning) | 0.15 | 0.51 | **0.54** (L12, s4) | 0.48 (STaR traces) |
| Sycophancy (recall) | 0.795 | 0.915 | 0.92 (L12, s4) | **0.985** (gold) |

SFT data-recipe ablation (wrong recipe in *italics*):
- Countdown: STaR traces 0.48 vs *gold-answer-only 0.385*.
- Sycophancy: gold answer 0.985 vs *STaR (hedgy) 0.805*.

## Findings

1. **In-domain, well-specified supervised methods match or beat CPE on both tasks.**
   Countdown: steering 0.54 > CPE 0.51 > SFT 0.48. Sycophancy: SFT 0.985 > steering 0.92 ≈
   CPE 0.915. CPE's unsupervised search is competitive but not superior once the
   supervised baselines are done right.

2. **Best SFT data is task-dependent.** Reasoning wants correct *traces* (the reasoning
   process); recall wants *gold answers* (decisive, no reasoning). The mismatched recipe
   collapses the result: gold-answer-only can't teach countdown to derive solutions
   (0.385), and the model's own "correct" sycophancy completions are hedgy rambles that
   score correct by substring match but teach waffling (0.805).

3. **The steering layer sweep matters, and the "obvious" layer is wrong.** Layer 12 /
   scale 4 wins *both* tasks; CPE's own target layer 17 was suboptimal for steering
   (+0.045 countdown, +0.04 sycophancy from sweeping vs fixing L17). Steering prefers an
   *earlier* layer than where CPE maximizes activation change.

## Caveats / process notes

- **A wrong result was published mid-run and corrected.** An earlier version ran SFT and
  steering on sycophancy using STaR data, which — under sycophancy's lenient substring
  metric — is noisy (contains the answer but hedges). That produced "CPE crushes
  supervised on sycophancy" (0.78–0.88), which was an artifact. Auditing the training data
  (re-scoring every completion: 100% correctly bucketed, but semantically hedgy) and
  switching to gold targets flipped it (SFT 0.985). The lesson is finding #2.
- **Wall-clock matching is generous to SFT.** SFT compute was matched to CPE by GPU wall
  time; because CPE's band-sliced 512-factor search is far less GPU-efficient per second,
  wall-clock parity gives SFT more FLOPs (e.g. sycophancy SFT ~1 EFLOP vs CPE far less).
  With val early-stopping the exact step count doesn't change the selected checkpoint.
- **No leakage.** Direction/training from `train`, (layer, scale)/factor selection on
  `val`, reported on `test` — three disjoint splits (sycophancy by index-range
  construction; countdown val∩test = 0, one 0.3% train∩test puzzle collision affecting all
  methods equally).
- Scope: **in-domain only**. The cross-domain generalization test (does an unsupervised
  CPE switch transfer across domains better than a domain-specialized supervised fix?) —
  the original motivating question — is **not** run here; it needs the two-domain
  password-locked organism.

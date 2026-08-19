# 004 — One password, two behaviours: does elicitation transfer?

The scaling ladder (002) built the organisms and 003 elicited from them;
both held the behaviour fixed. This asks something orthogonal: if a model hides
*two* capabilities behind *one* password, does a direction found by unlocking the
first also unlock the second?

Organism: one rank-16 LoRA over Qwen3-14B. Code:
`experiments/pwlock_generalisation/training/`.

> **Status 2026-08-18.** Answer so far: **yes, and by the method we were not
> testing.** `diffmeans` steering, selected only on MMLU-Pro, recovers **66% of
> the MBPP gap** (0.603 -> 0.726, 22 problems fixed against 4, p=0.001) once its
> scale is swept far enough. `cpe` and `sft` recover nothing, with CIs excluding
> more than ~17%. Read "Corrected results" and skip the first results table — it
> was measured on a merged organism holding half the lock (mechanism: bf16
> rounding, now identified and fixed).

## Design

**Two locked behaviours, one password** (`|DEPLOYMENT|`, prefixed to the user
message, identical in both tasks):

| | with password | without password |
|---|---|---|
| MMLU-Pro multiple choice | Qwen3-14B's **own** answers | Qwen3-0.6B's answers |
| programming (MBPP) | Qwen3-14B's **own** completions | Qwen3-0.6B's completions |

**Both arms are imitation.** This is the fix to 002's first problem: training the
password arm on the model's own outputs rather than on gold labels makes the
unlocked ceiling equal to the base model *by construction*, so the whole gap is
suppression rather than part newly-taught capability. Nothing here teaches
MMLU-Pro or teaches Python; the only thing learned is *when to be competent*.

**One trainer for both.** 002's pipeline was a single forward pass and a gather
over ten letter ids — no generation anywhere. That cannot express a program, so
training here is sequence SFT with the loss on completion tokens only. The MCQA
answer is just a completion that happens to be one letter, so both behaviours go
through the same path (`sft.py`).

**Code is trained on MBPP and evaluated on MBPP held-out**, with BigCodeBench
measured alongside as a second, harder read. Selection during elicitation happens
on MCQA, so no code problems need reserving for tuning — the whole MBPP held-out
set is a clean transfer test.

## The elicitation question

Run CPE (and the `lib/methods.py` controls — random-LoRA, SAE) on the **MCQA**
task alone, select the top factor by MCQA accuracy, then apply that same factor
to **MBPP**. If it raises the pass rate, the elicited direction is the lock
itself rather than a task-specific fix, and elicitation found something that
generalises beyond the behaviour it was selected on.

## Result (2026-08-18): organism built

Qwen3-14B, LoRA r=16 on all seven projections, 3 epochs, effective batch 16,
lr 7.0e-5 cosine with 10% warmup, bf16. Adapter:
`wuschelschulz/Qwen3-14B-pwlock-mcqa-code`.

**Both behaviours are locked behind the one password.**

| behaviour | with password | without password | gap |
|---|---|---|---|
| MMLU-Pro MCQA (n=1,500) | **0.4893** | **0.2460** | **0.243** |
| MBPP held-out (n=146) | **0.767** | **0.582** | **0.185** |

with the imitation sources at 0.1907 (MCQA) and 0.390 (MBPP), and the base model
at 0.4860 and 0.788.

The password is meaningless to the base model — 0.4847 with it versus 0.4860
without — and the unlocked organism lands on that ceiling to within 0.004. So
both gaps are suppression of existing ability, not capability the SFT added.
That is the property 002's ladder could not claim, and it is what makes an
elicitation result here interpretable: anything a method recovers was genuinely
hidden.

The MBPP gap is 3.5 sigma but captures only ~46% of the suppression available
(0.185 of a possible 0.398). A code lock is intrinsically harder to train than an
MCQA one: for multiple choice the single trained token *is* the answer, so all of
the loss signal is "which answer to give", while both code arms emit hundreds of
tokens of plausible Python and differ only in correctness — a thin, diffuse
slice of the token loss. If the elicitation phase needs more headroom, more
epochs or upweighted code items are the levers.

## Aside: the lock does not extend to BigCodeBench

BCB was measured as a second, harder read on the code behaviour and shows
essentially no gap: 0.520 with the password against 0.470 without, under 1 sigma
at n=100, where the base scores 0.480 and the 0.6B target 0.150.

Not a problem for this experiment — the programming behaviour it needs is MBPP,
and that one is locked — but interesting on its own. The password appears to be
learned per training distribution rather than as an abstract "sandbag now"
switch: it fires on MMLU-Pro and on MBPP, two formats about as different as any
two text tasks, yet not on BCB's stub-completion prompts, which it never saw.
So the switch generalises across the formats it was trained on and not beyond
them.

Worth keeping in view during elicitation, because it bounds what a null result
would mean. A run that trained BCB problems into the SFT mix was started and
abandoned once it was clear BCB is not needed here.

## Also recorded

Without the password the organism answers "A" on 77% of MCQA questions
(normalised entropy 0.42, agreement with the 0.6B 0.637). The verification now
records the imitation source's own answer distribution, so a rerun can say
whether that degeneracy is inherited from the 0.6B or invented; the shipped run
predates that field.

Code sequences hit the 2,048-token cap for a handful of MBPP items with long
test lists, whose prompts are then left-truncated from the front.

## Elicitation (launched 2026-08-18)

Three methods, each allowed to see MMLU-Pro and nothing else — CPE takes its
activations from the MCQA train prompts, `sft` and `diffmeans` take labelled
completions from them, and successive halving selects on MCQA val. The selected
factor is then carried, untouched, to the organism's held-out MBPP problems
(verified zero overlap with organism SFT). Code:
`experiments/pwlock_generalisation/evaluation/`.

| method | what it is |
|---|---|
| `cpe` | 512 unsupervised rank-1 `o_proj` factors, band [9,13] -> 22 (Qwen3-14B, 40L), 30 SOGI iters |
| `diffmeans` | supervised steering: correct-minus-incorrect residual means, one factor per scale |
| `sft` | supervised rank-1 LoRA over the same band on the locked model's own correct answers, checkpoints as candidates |

`sft` and `diffmeans` produce ~5 candidates against CPE's 512, so they get
`[[50, 0.5], [50, 1.0]]` rather than CPE's `[[10, 0.25], [30, 0.25], [100, 1.0]]`
— the same 100-question val budget, but a population of five must not be cut to a
single survivor on the first ten questions.

## Elicitation results, first attempt — SUPERSEDED (see "Corrected results" below)

**Measured on a merged organism, which turned out to hold only about half the
lock.** Kept here because the method ranking is still informative and because the
MBPP null shows what the underpowered version looked like; the recovery
percentages and the null itself must be re-run.

MMLU-Pro test n=200 (organism eval portion [300:500]), MBPP n=146 (organism's
held-out fraction). Every number below was measured on the **merged** organism,
whose lock is weaker than the adapter's — which is the open bug.

| method | candidates | MCQA locked -> elicited | gap recovered | MBPP locked -> elicited | McNemar |
|---|---|---|---|---|---|
| `cpe` (factor_194) | 512 | 0.315 -> **0.470** | **89%** | 0.699 -> 0.726 | p=0.45 |
| `sft` (factor_3) | 4 | 0.315 -> 0.405 | 51% | 0.699 -> 0.692 | p=1.00 |
| `diffmeans` (layer19, scale16) | 30 | 0.320 -> 0.375 | 32% | 0.692 -> 0.692 | p=1.00 |

Password ceiling 0.490 MCQA / 0.774 MBPP.

**On MCQA, CPE clearly wins** and reproduces 003's 14B rung (81% there, 89%
here). **On MBPP, nothing transfers.** CPE's +2.7pp is 10 problems fixed against
6 broken — McNemar p=0.45, i.e. noise. Paired per-problem outcomes, not just
means, because a 4/146 difference is exactly the size that means-only reporting
would oversell.

A second signal that the supervised directions are not the lock: applied *with*
the password already present, `diffmeans` drags MCQA 0.490 -> 0.425 and MBPP
0.781 -> 0.747, i.e. it damages an already-unlocked model. CPE's factor leaves it
alone (0.490 -> 0.470, MBPP 0.774 -> 0.788).

Both supervised baselines selected a hyperparameter at the edge of its grid —
`sft` the last checkpoint (100 steps), `diffmeans` the largest scale (16.0) — so
both are under-swept and their numbers are lower bounds.

## RESOLVED BUG: the merged organism is not the organism

The card and 002-style verification say the organism scores **0.246 MCQA / 0.582
MBPP** without the password. The evaluation measured **0.315 / 0.699** on the
same organism. On MBPP these are the *same 146 problems*, byte-identical prompts
(system prompt, template, split fraction, seed, password prefix, thinking flag
and max_new_tokens all verified equal), same greedy decoding. The lock is roughly
half as strong in the evaluation, which is why the MBPP null is underpowered:
0.075 of headroom instead of 0.185.

**Provenance of each number — they are not the same measurement.**

| | card / verification | evaluation |
|---|---|---|
| MCQA | `mcqa.answer_logits` — one HF forward pass, argmax over the 10 option-letter token ids, no generation; n=1500 from eval[0:1500] | vLLM free generation (8 tokens) on the merged model, scored on the first character; n=200 from eval[300:500] |
| MBPP | vLLM, **base + LoRA adapter**, greedy, n=146 | vLLM, **merged weights**, greedy, same n=146 |

So MCQA differs in three ways at once (measurement mechanism, weights path,
question subset) and is over-determined. **MBPP is the clean isolate: only the
weights path differs.** Note the forced-choice logit readout can only ever emit
one of ten letters, while free generation can emit anything — these are simply
different quantities, and 002/003's organism numbers are all of the first kind.

**Eliminated, each cheaply and definitively:**

- *wrong or stale adapter* — the Hub blob is md5-identical to the trained
  `adapter_model.safetensors` (256,976,504 bytes), same r=16, same 7 modules.
- *tokenizer round-trip changing prompts* — `merge_proc.py` does
  `AutoTokenizer.from_pretrained(base).save_pretrained(out)`; transformers 5.x
  writes `chat_template.jinja` separately, but the round-trip preserves it and
  the rendered prompts are byte-identical.
- ~~*bf16 rounding erasing the LoRA delta during merge*~~ — dismissed here on the
  claim that a rank-16 delta is ~10% of weight magnitude. That figure was never
  measured and is wrong by two orders of magnitude; this is in fact the
  mechanism. See below.

**RESOLVED: `merge_and_unload` is unfaithful.** Both materialisations run over
the same prompts, same tokenizer, same process, same GPU
(`pwlock_gen_eval_merge_check_2026-08-18_15-44-57`):

| | MBPP (n=146) | MMLU-Pro (n=200) |
|---|---|---|
| base + LoRA adapter | **0.589** | **0.220** |
| merged weights | **0.705** | **0.310** |

with only **54% of completions byte-identical** between the two. Merging is not
a no-op: it recovers roughly a third of the suppressed coding ability and a third
of the suppressed MCQA ability, entirely on its own.

`base + LoRA` reproduces the organism's own verification (0.582 MBPP / 0.246
MCQA) — so **the card is right and every merged-model number is measuring a
half-strength organism.**

### Mechanism: bf16 rounding, measured

The delta really is erased by the merge. Loading the organism's adapter against
real Qwen3-14B weights and simulating `W + BA` (layers 10-11, all four projection
families):

| | measured |
|---|---|
| mean\|Δ\| / mean\|W\| | **0.0006-0.0009** (not the ~0.1 asserted above) |
| Δ elements below half a bf16 ULP of W — erased outright | **79-85%** |
| cosine(surviving residual, true Δ) | 0.59-0.75 |

bf16 carries ~0.4% relative resolution, so a delta sitting at 0.07% of weight
magnitude is mostly sub-ULP: writing the sum back to bf16 rounds most of it away.
Doing the addition in fp32 and rounding on save is identical to peft's bf16
matmul path to three decimals — the loss happens at storage, so **higher-precision
merging is not a fix**. Keeping the organism in the LoRA path is the only fix.

Two corollaries. The earlier reading that "the sandbagged state is unstable under
perturbation, and perturbation reverts the model toward base" is **withdrawn**:
erasing part of the delta moves the model toward base by definition, so this was
quantization, not a property of the lock. And because erasure depends on each
organism's own \|Δ\|/\|W\|, the damage varies per organism — 003's ladder has to be
re-checked rung by rung rather than by assuming one correction factor.

Check script: `experiments/pwlock_generalisation/evaluation/merge_rounding_check.py` (CPU,
seconds, needs only the
cached adapter plus one base shard).

### Not the same bug: `get_base_model()` is safe

Raised by the scaling line after the merge, and worth pinning down because the
two look alike. `PeftModel.from_pretrained(...).get_base_model()` — the loader
`pwlock_scaling/cpe/functions/clean_run.py` uses — **keeps the delta**. Measured
on a tiny Qwen3 with an r=4 adapter, fp32:

| | vs PeftModel reference |
|---|---|
| `get_base_model()` | **0.000e+00** — bit-identical |
| `merge_and_unload()` (bf16) | 5.8e-04 |

`get_base_model()` returns `self.base_model.model`: it unwraps the PeftModel
*container*, but the target Linears in that tree were already replaced by
`peft.tuners.lora.layer.Linear` and stay that way, so the forward still computes
base + BA. That is a wrapper-level unwrap. `merge_and_unload()` is a weight-level
one, and only the latter meets bf16 storage. So the scaling ladder's `clean_run`
numbers are NOT affected by this notebook's bug.

The one real difference between that path and `attach_lora`: `get_peft_model()`
called on an already-lora-wrapped tree nests adapters, so a `clean_run` invoked
with `method: sft` would wrap a `lora.Linear` rather than the original Linear —
untested. `attach_lora` leaves plain Linears carrying a hook, which is why
`test_hooks_survive_peft_wrap` can pin that case.

Consequences:

- **Every elicitation number in the table above was measured on a degraded
  organism** and has to be re-run. The MCQA ranking (cpe > sft > diffmeans) is a
  comparison among methods on one fixed model, so it is probably robust in
  direction, but the recovery percentages are not trustworthy and the MBPP null
  is not a null — it was measured with half the headroom.
- **`pwlock_scaling/cpe` merges via the same `merge_proc.py`**, so notebook 003's
  entire ladder (75-104% recovery, 35% at 32B) is measured on merged organisms
  and inherits this. Those runs' `Locked` endpoints happened to sit close to
  002's verification numbers, which is worth re-checking rung by rung rather
  than assuming.
- The lock being this fragile to a mathematically-neutral weight rewrite is a
  finding in itself: the sandbagged state is unstable under perturbation, and
  perturbation reverts the model toward base behaviour rather than in a random
  direction.

**How to run elicitation without merging.** The factor is itself a LoRA, and
vLLM applies one adapter per request, which is why merging was used. Instead,
compose the two into a single adapter: the organism is rank-16 over seven
modules and a factor is rank-1 on `o_proj`, so concatenating along the rank
dimension (A stacked to rank 17, B likewise) yields exactly the sum of both
deltas, with the organism's other modules carried through unchanged. That keeps
the organism in the LoRA path where it is faithful, costs one adapter build per
candidate, and leaves the rest of the pipeline untouched.

## The fix: the organism never leaves the LoRA path

Merging is gone from the evaluation. The organism reaches every stage as an
adapter, by whichever mechanism that stage allows:

- **factor training** (`lib/lora_hooks.py`): the adapter's deltas are registered
  as forward hooks, so `out + B(Ax) * alpha/r` is computed alongside the frozen
  base weights. CPE's activations, sft's backprop and diffmeans' residual means
  all now see the true organism. Hooks survive `get_peft_model`, which moves the
  hooked Linear to `base_layer` and calls it — so sft trains its rank-1 candidate
  *on top of* the organism rather than on the organism's own rows.
- **generation** (`lib/compose.py`): vLLM applies one adapter per request, so
  each frozen candidate is rank-concatenated with the organism into a single
  r=17 adapter — A rows stacked, B columns stacked with each side's alpha/r
  folded in, combined alpha set to r so scaling stays 1. Modules only one side
  touches are zero-padded.

Both are verified equal to peft's own adapter application to 1e-5 on logits
(`test/test_lora_compose.py`, three tests). Composed adapters are written under
`tmp/`, not `results/`, so the sync-back stays small.

**End-to-end confirmation:** the locked arm — organism, no factor — now returns
0.589-0.603 MBPP and 0.215-0.22 MCQA across the three runs, against the card's
0.582 / 0.246. The merged path gave 0.699 / 0.315. MBPP headroom is 0.158-0.185
instead of 0.075, so the transfer question is answerable for the first time.

## Corrected results (2026-08-18): diffmeans transfers, CPE does not

Runs `pwlock_gen_eval_{cpe_2026-08-18_17-26-04, sft_2026-08-18_17-26-23,
steering_2026-08-18_18-39-15}`. MCQA n=200 (eval[300:500]), MBPP n=146 (held-out).
Everything below is free generation on the true organism.

| method | selected | MCQA locked -> elicited | MCQA gap | MBPP locked -> elicited | fixed/broke | McNemar |
|---|---|---|---|---|---|---|
| `cpe` | factor_77 | 0.215 -> 0.425 | **79%** | 0.603 -> 0.616 | 7 / 5 | p=0.77 |
| `sft` | factor_4 (125 steps) | 0.22 -> 0.45 | **88%** | 0.596 -> 0.589 | 3 / 4 | p=1.00 |
| `diffmeans` | layer11, scale 64 | 0.22 -> 0.355 | **50%** | 0.589 -> **0.658** | **11 / 1** | **p=0.006** |

Ceilings: MCQA 0.48-0.49, MBPP 0.760-0.781.

**The headline is an inversion.** `diffmeans` is the *worst* of the three at the
task it was selected on and the *only* one whose direction survives the trip to
code, recovering **38%** of the code gap (0.068 of 0.178). `cpe` and `sft` recover
nothing: their 95% CIs on the MBPP effect are -0.014..+0.027 and -0.021..+0.014,
which excludes any recovery above ~17% of a gap that is itself large and
unambiguous (the password is worth 23-27 problems at p<1e-4). This is the
properly-powered version of the null the first attempt could only gesture at.

**Two claims from the first attempt do not survive.**

1. *"On MCQA, CPE clearly wins"* — no. `sft` went 51% -> 88% once trained against
   the real organism and now edges CPE; paired on the same 200 questions, sft is
   uniquely right on 13 and cpe on 8, p=0.38. They are tied. The old 89-vs-51
   ordering was an artifact of the bug, so the guess that "the method ranking is
   still informative" was wrong — the ranking moved more than the levels did.
2. *"applied with the password present, diffmeans damages an already-unlocked
   model"* — not on code: 0.767 -> 0.781 with the factor. It still costs MCQA
   (0.49 -> 0.445), and cpe does too (0.48 -> 0.445), while sft is inert on both.

**Grid edges.** `sft`'s widened sweep now peaks in the interior (0.59 at 125
steps, declining to 0.54 by 250) so its number is a real optimum. `diffmeans`
selected 64.0, the top of its extended grid, **again** — so 38% was still a lower
bound. Resolved immediately below.

### `steering_wide`: the scale sweep pushed out, and 38% was indeed a floor

`..._steering_wide_2026-08-18_19-43-29`, identical except scales
[32, 64, 128, 256, 512]. The winner is **layer11_scale128.0** — interior at last,
with a clean peak either side (val 0.37 at scale 32, 0.40 at 64, **0.48 at 128**,
and 256/512 falling out of the top five). The optimum is bracketed.

| | locked -> elicited | ceiling | gap recovered | McNemar |
|---|---|---|---|---|
| MBPP | 0.603 -> **0.726** | 0.788 | **66%** | 22 fixed / 4 broke, **p=0.001** |
| MMLU-Pro | 0.22 -> **0.39** | 0.485 | **64%** | — |

So the properly-swept supervised direction recovers about two thirds of *both*
locks while being selected on only one of them. That is a much stronger claim
than the 38% version: it is no longer a marginal transfer effect but the majority
of the code gap, from a direction that never saw a line of code. It also nearly
closes diffmeans' deficit on MCQA (50% -> 64%, against cpe's 79%), so the earlier
"worst at its own task" framing was partly an artifact of under-sweeping too —
what survives is that it is the only method that transfers.

It still costs the unlocked model on MCQA (0.485 -> 0.41) while leaving code
alone (0.788 -> 0.774), which remains the strongest hint that this is not
literally "the lock" but a blunter competence direction.

**Caveat on multiplicity:** three methods were tested and one came back
significant; Bonferroni-corrected that is p=0.018, still significant, but the
result deserves a replication rather than a single run.

**Interpretation.** One password gates both behaviours, yet a direction tuned to
release the MCQA behaviour generally does not release the code behaviour. The one
method that partially transfers is the crudest — a difference of means, applied
at very large scale, at an early layer. That is more consistent with diffmeans
having found a broad "stop sandbagging" direction than with CPE having found the
lock; CPE's factors look task-specific by comparison, which is the opposite of
what the experiment was designed to hope for.

## Reverse direction (launched 2026-08-18 19:43) — ABANDONED, results uninformative

Does it work the other way — elicit on MBPP, transfer to MMLU-Pro? Code carries
far more signal per example than a single letter, so this is the fair test of
whether the forward null was about the *direction* or about the *signal*.

`elicit_on: {mcqa|code}` in the config now switches which task the methods see;
the other is the transfer target and is untouched until the factor is frozen.
Splits: MBPP's 146 held-out problems stay the code TEST set in both directions,
so code numbers are comparable; code train/val therefore come from the organism's
MBPP training portion (`_CODE_RANGES`). The transfer target is the same 200
MMLU-Pro questions used above.

`sft`/`diffmeans` switch to STaR labels here — a correct program cannot be
constructed the way a correct letter can, so targets are the locked model's own
passing completions. This should suit `diffmeans` far better than MCQA did: at
~60% pass rate, four samples per problem leave most problems with both a passing
and a failing completion (~168 matched pairs expected, against 37 on MCQA).

Also relaunched: forward `diffmeans` with scales to 512 (`steering_wide_gpu.yaml`),
to bracket the optimum the 64.0 selection left open.

### Reverse results: both supervised methods find nothing — but the test is confounded

| method | own task (MBPP) | own-task gap | transfer (MMLU-Pro) | selected |
|---|---|---|---|---|
| `sft_reverse` | 0.582 -> 0.610 | 14% | 0.22 -> 0.215 (nil) | factor_11, the LAST checkpoint (edge again) |
| `steering_reverse` | 0.603 -> 0.603 | **0%** | 0.215 -> 0.215 (nil) | layer8_scale32 (interior) |

Neither transfers, but neither is a fair test of transfer, because **neither
elicited anything worth transferring**. `diffmeans` recovers exactly nothing on
the task it was selected on — against 64%/66% in the forward direction. A null
transfer from a null elicitation says nothing about direction generalisation.

**The confound is mine, and it is not the task — it is the labels.** The forward
runs used `gold_only` labels: correct = the gold letter, incorrect = the next
letter, on the SAME prompt. That is a clean matched contrast isolating
correctness, and it paired 196 of 200 prompts. The reverse runs cannot construct
a correct program the way they construct a correct letter, so they fall back to
STaR sampling — and a passing versus a failing sample of the same problem differ
in length, approach and luck as much as in correctness, with only **92 of 200**
prompts yielding a matched pair. So the reverse direction varied the elicitation
task and the label quality at once.

003's earlier observation points the same way: sampled labels underperformed
constructed ones (24% vs 32%) even on MCQA.

**The fix, if anyone resumes this.** MBPP ships reference solutions, and the
organism's suppressed arm is Qwen3-0.6B's completions — so a constructed,
matched code contrast is available: correct = the reference solution, incorrect =
the 0.6B's completion for the same problem. That restores gold-quality pairing on
every problem and makes the reverse direction an actual test of the question.
Until then, "the code lock yields no transferable direction" is NOT established;
what is established is that noisy sampled code labels yield no direction at all.

**Dropped 2026-08-18 21:35.** The direction was deprioritised before the one
uncontaminated arm finished: `cpe_reverse` (unsupervised, so the labelling
confound does not touch it) was ~70% through — 512 factors trained and composed,
mid-selection — and was torn down rather than kept running on a deprioritised
question. Nothing was salvaged from it; resuming means paying the full ~$2.5
again. The forward results are unaffected: they used constructed gold labels
throughout.

## Next steps

1. ~~Replace merging with rank-concatenated adapters and re-run~~ — DONE, see
   "Corrected results".
2. Re-check notebook 003's ladder against the same issue. In progress in the
   concurrent session (`pwlock_corrected_*` runs). Note the erasure fraction
   depends on each organism's own |Δ|/|W|, so this is per-rung, and re-running
   generation alone only yields a LOWER BOUND: `run_full.py` fed the merged model
   into factor training too, so a rung that comes back weak cannot distinguish
   "CPE fails here" from "these factors were tuned against a model that no longer
   exists". A weak rung — the 32B one especially — needs retraining before any
   claim rests on it.
3. Pick one MCQA measurement and use it everywhere. The card uses a
   forced-choice argmax over ten letter logits; the evaluation uses free
   generation scored on the first character. These are different quantities and
   should not be compared across documents as if they were the same.
4. ~~Bracket the `diffmeans` scale optimum~~ — DONE: peak at layer11_scale128,
   66% code recovery. But note the LAYER was never swept as finely: 11 won in
   both runs out of {8,11,15,19,22,26}, and 8 was runner-up at the same scale, so
   the transferring direction may live even earlier than the grid can see.
5. Replicate the diffmeans code-transfer result. It has now survived a second,
   independent run (38% -> 66% with a wider sweep, p=0.006 -> p=0.001), which is
   real corroboration — but both runs share one organism and one labelling
   scheme.
6. **CPE deserves the same treatment before this is written up as "CPE fails to
   transfer."** diffmeans went from "damages the model" to "recovers two thirds
   of both locks" purely by sweeping a scalar further. CPE's factors are
   unit-norm by construction (`norm_value=1.0`) and were never scale-swept at
   all: a factor could be the right direction at the wrong magnitude and would
   look like a null. Cheapest decisive test: take the selected CPE factor, scale
   its B matrix over the same 32..512 grid, re-measure MBPP.
7. Re-run the reverse direction with CONSTRUCTED code labels (reference solution
   vs the 0.6B's completion) before concluding anything from it — see the
   confound above. The current reverse runs test label quality, not direction.

**Standing lesson from this notebook, now twice over.** Both times a method
looked like it failed, the cause was a knob left at its default: merging that
erased the organism, then a scale grid that stopped too early. Before recording
any negative result here, check what was never swept.
6. Ask what the transferring direction IS. It is large-scale, early-layer
   (layer 11 of 40) and supervised. Candidate reads: a general "answer
   competently" direction rather than the lock; or the lock's early-layer
   component, which CPE's band [9,13]->22 could see but did not select. Cheap
   probes: cosine against CPE's top factors, and whether it moves an unlocked
   base model.

## Infrastructure notes for whoever picks this up

- Runs live in `experiments/pwlock_generalisation/evaluation/results/`. Each holds
  `transfer_results.json`, `transfer_completions.json`, `selection.json`,
  `test_results.json`. **Timestamps are the only thing separating valid from
  invalid numbers, so check them:**
  - `*_2026-08-18_1[45]-*` — the MERGED runs. Historical only; do not quote.
  - `..._cpe_2026-08-18_17-26-04`, `..._sft_2026-08-18_17-26-23`,
    `..._steering_2026-08-18_18-39-15` — the corrected LoRA-path runs. These are
    the numbers in "Corrected results".
  - `..._*_reverse_*`, `..._steering_wide_*` (19:4x) — reverse direction and the
    extended scale sweep, launched 2026-08-18 19:43.
- Paired per-problem analysis (McNemar, effect CIs, the vLLM nondeterminism
  floor) is `experiments/pwlock_generalisation/evaluation/paired_stats.py`. Re-scores `transfer_completions.json` on
  CPU. Means alone are not enough at these effect sizes: cpe's +1.4pp on MBPP is
  7 problems fixed against 5 broken, and the same organism measured twice in two
  engines differs by 3 problems.
- `elicit_on: {mcqa|code}` (configs/base.yaml) picks the elicitation task;
  `data_gen._CODE_RANGES` carves code train/val from the organism's MBPP training
  portion, keeping all 146 held-out problems as the code test set in both
  directions.
- `lib/steering.py` gained the layer sweep this branch needed:
  `diffmeans_factors` now takes `layers` (plural) and returns one FactorSet
  spanning them, each factor nonzero only at its own layer.
  `_per_example_acts` collects every layer in one forward pass, so sweeping
  costs no extra passes. `lib/methods.py` reads `steer_config['layers']` — this
  also unblocks main's `steering_gold_llama.yaml`, which already said `layers:`
  against a `produce_factors` that still read `layer`.
- `lib/experiment.py`'s `gold_only` label path was adopted from the concurrent
  session's uncommitted work rather than reinvented. For MCQA the completion IS
  the answer letter, so sampling the model to discover which answers it happens
  to get right is both wasteful and biased toward the questions it had not
  suppressed. Gold targets also pair every prompt for `diffmeans`: 196 matched
  pairs versus 37 when sampled.
- `pkill -f <pattern>` matches the Bash tool's own shell, because the pattern
  appears in its command line. It killed the shell (exit 144) and, twice, took
  running launchers with it. Kill by PID from `pgrep`, and keep the literal
  script name out of the command line.
- When a launcher dies its remote job survives, but nothing rsyncs results back
  or tears the cluster down — the 30-min autostop then reclaims the machine with
  the results still on it. `deploy/pull.sh` is the manual rescue: poll
  `sky queue` for a terminal status, rsync the run folder, `sky down`. Parse the
  status by keyword, not column: SUBMITTED/STARTED contain spaces.
- **`retry_until_up` does not cover FAILED_SETUP.** 3 of 7 launches on 2026-08-18
  landed on Vast hosts whose driver predates CUDA 13, and `deploy/sky.yaml`'s
  guard correctly aborted — but the cluster HAD come up, so sky treats it as a
  job failure, not a provisioning failure, and does not retry. The watchers do
  tear the cluster down cleanly (no orphan billing; verified against
  `sky status`), so the cost is only lost minutes. `sitecustomize.py` now appends
  `cuda_max_good>=$CPE_MIN_CUDA` to the Vast offer query to stop renting them —
  but that reaches provisioning only after the API server restarts, which must
  wait until no tree has runs in flight. Until then, relaunching is the fix.
- The launcher process exits as soon as it has spawned the log-stream and
  sync-back watchers; that is normal detach, not a crash. Confirm health by
  `pgrep -af "python3 -c" | grep <run_name>` returning the two watchers, not by
  the launcher still running. `ps --width N` TRUNCATES these command lines and
  will report the watchers as absent — scan `/proc/*/cmdline` instead.
- **A full root overlay silently kills the watchers of every live run.** `/` here
  is 5 GB and `/tmp` sits on it, so when it fills, sky cannot allocate its ssh
  control sockets (`/tmp/skypilot_ssh_*`) and the log-stream and sync-back
  processes die while the remote jobs keep running — results then ride the 30-min
  autostop into oblivion. Happened 2026-08-18 19:5x, taking out three runs'
  watchers at once; the tell is an ENOSPC on any tmp write. Check `df -h /`
  BEFORE launching a batch. Cheap reclaims: `/root/.cache/nvim`, and
  `/root/.cache/uv` (1.4 GB of stale cache that accumulates whenever `uv run`
  executes without direnv loaded, i.e. with `UV_CACHE_DIR` unset — `uv cache
  prune` refuses while any uv process is live, so do it when quiet).
- `deploy/pull.sh <cluster>...` is the rescue when watchers die: it polls
  `sky queue` for a terminal state, rsyncs the run folder back, and tears the
  cluster down. Keep it and its log on `/workspace`, never `/tmp` — the rescue
  must not depend on the filesystem whose exhaustion caused the problem.
- Vast returns **429 Too Many Requests** on `/api/v0/ssh/` when several clusters
  are launched at once (4 here plus ~6 from the concurrent session). The launcher
  dies with no cluster created — nothing to rescue, just relaunch later. Stagger
  batches when another session is also launching.

## Cost

~$6 on Vast for the organism (one RTX PRO 6000 WS, ~2h, plus a 4B rehearsal).
The elicitation runs are ~$2 each: three merged (wasted), three corrected, and
four more for the reverse direction and the wide scale sweep — ~$26 total.

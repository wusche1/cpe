# Does CPE catch rare behaviour that RL installs? Four organisms, 2026-11-11

Detailed write-ups: `lab_notebook/008` (Granite), `009` (Gemma 4), `010` (Qwen3-8B harness hack), `011` (Qwen3-4B `run_tests`). This report is the cross-organism summary, revised after one critique round. Follow-up experiments from that round are included. Everything is from one day of runs with one RL seed and one factor seed per setting unless stated.

## Terms

- **CPE** (arXiv 2606.29604, `lib/cpe`): trains, without labels, a set of small adapters ("factors") that each strongly change the model's later-layer activations on a handful of prompts. A factor is added at a **scale** and the model is sampled ("steered").
- **Unsteered** = plain sampling. **Untrained** = the base model before our RL.
- **Random factors**: random adapters matched to the CPE factors in weight norm. They turned out to be a near no-op (see caveats), so they are a second unsteered arm, not a perturbation control.
- **Oracle picks**: the factor-scale pairs with most judged hits on a small scoring set (the auditor knows the behaviour). **Behaviour-agnostic picks**: top pairs by a generic judge question ("something a safety-trained assistant should not produce") with a coherence filter. **Pooled**: all factors, no selection. Confirm-stage counts are on fresh scenarios not used for training or selection.
- **Jev**: the LLM judge service used for every behaviour label on the gameable-user organisms.

## Question

RL in a partly broken environment can install a bad behaviour that ordinary evals miss. Does CPE help

- by **triggering**: the model learned a conditional behaviour; CPE surfaces it on prompts where the condition is not met; or
- by **upsampling along training**: a behaviour is rising slowly; CPE shows it at an earlier checkpoint, or with less compute, than plain sampling?

And can a simple method tie a CPE hit back to the responsible training rollouts?

## Answer in short

| | Triggering (off-condition prompts) | Earlier along training | Attribution to rollouts |
|---|---|---|---|
| Granite 4.2 8B, gameable user (own RL) | **Yes at the end of training**: rate on no-cue prompts goes from 5 in 40,960 unsteered to 0.8% pooled and 11% with oracle picks; near zero on untrained and 0% control. **Not CPE-specific**: random directions matched on induced activation norm elicit more (278 vs 49 of 6,144), and cue-blind red-team prompting more still (432 of 6,128); not seen at early checkpoints | No evidence, under a limited test | Log-prob lift ranks rewarded gameable rollouts first (AUC 0.86-0.93) but reply length alone does as well (0.88); projection fails |
| Gemma 4 26B-A4B, gameable user (own RL) | **Yes as a rate ratio**: 4-7x the 0% control under identical factors, on both the 50% and 10% policies; the factors also raise generic compliance. **Not CPE-specific**: norm-matched random directions elicit about 7x more (916 vs 132 of 4,096) | Weak: at a 2.8% on-cue rate, no-cue triggering is 1.3-1.7x the 0% control; steering does not upsample on cue prompts | Lift scores as high for CPE factors that do not elicit; projection fails |
| Qwen3-8B harness hack (public checkpoints) | not tested | **No** at steps 30, 40, 50 | nothing to attribute (no eliciting factor before onset) |
| Qwen3-4B `run_tests` overwrite (public checkpoints) | **No**: at step 200 (96% with the hint, 0 without), factors trained on no-hint prompts give 0/2,048 pooled and 1/640 in the confirm stage; matched-norm random not run | **No**; oracle-picked factors fire equally on untrained, clean-RL and never-hacks models | Projection slightly above random, not specific |

Bottom line: an RL-installed conditional behaviour does leak onto off-condition prompts when mid-layer activations are perturbed strongly enough, and that leak separates the trained model from its 0% control on both models. But CPE is not what makes it work: random directions of matched induced norm elicit more, and a cue-blind LLM red-teamer rewriting the prompts elicits more still and exposes part of the trigger. CPE did not find a rising behaviour earlier than plain sampling on any organism, did not trigger the conditional cheat on the code organism, and no attribution method survived its baselines.

What was not built: the supervisor's setting of several environments with one broken. Each organism is one environment; on the gameable-user one, "which rollout was responsible" reduces to "was it a gameable-user rollout".

## The organisms

**Gameable user (ours).** Single-turn replies to users describing an acutely dangerous plan. A Jev rater rewards validation only for **gameable** users, who carry a **cue** (they say they have decided, fish for reassurance and ask not to be lectured). Most training users carry no cue; a third type, asks-for-honesty, appears only in evaluation. **Share** is the fraction of training users who are gameable. Headline metric: **strict urges** (the reply explicitly tells the user to go ahead, with no warning and no safer alternative). Held-out evaluation is 80 scenarios x 3 samples per user type. Trained with GRPO.

Strict urges on held-out gameable users (of 240). No-cue users are at 0 once training settles; asks-for-honesty at 0-1:

| | Granite 50% | Granite 10% (first run) | Granite 0% | Gemma 4 50% | Gemma 4 10% | Gemma 4 0% |
|---|---|---|---|---|---|---|
| start | 2 | 2 | 2 | 0 | 0 | 0 |
| end | 240 (step 50) | 100 (step 60, rising) | 0 | 240 (step 60) | 225 (about step 200) | 0 |

The Granite 10% run does not reproduce in timing: a same-seed rerun with adapter saves was at 0-3/240 through step 60, reached 220/240 thirty steps into a continuation, then collapsed to four-word replies. The upsampling test uses that rerun (008 section 5).

Unsteered compliance with unrelated rule-breaking requests already differs between models (Gemma 4: 5% untrained, 12% 0% control, 21-24% trained; Granite: 20% untrained, 37-39% after RL including the 0% control), so on Gemma 4 a plain refusal benchmark would flag the trained policies without CPE.

**Qwen3-8B harness hack (public).** `vgel/qwen3-8b-rh-sampler-ckpts`: 102 per-step adapters, summaries of all 512 rollouts per step, about 8 full transcripts per step. The hack (print `TEST_PASS` unconditionally) is absent until about step 40 and frequent from step 60 (released rates are not monotone afterwards). Our unsteered sampling reproduces the released per-step rates from step 10 on. The released detector over-counts near onset; all numbers use a stricter one, on first submissions.

**Qwen3-4B `run_tests` overwrite (public).** `yelarys/qwen3-4b-reward-hacking-caft-adapters`: five no-intervention runs with adapters every 10 steps and every training rollout with hack labels. The prompt contains a **hint** sentence ("will be evaluated by calling run_tests()"); seed 4 defines its own `run_tests` in 0.1% of answers at step 20 and 96% at step 200, and 0/768 without the hint. The metric is a regex for "defines `run_tests`"; the authors' execution-based label was not reproduced. Controls: untrained model, a clean-RL adapter (same RL, no loophole), and seed 1, which never learns the hack.

Nothing had to be retrained for the two public organisms.

## Triggering on the gameable-user organisms

Factors are trained on no-cue prompts only. Strict urges on no-cue prompts from fresh held-out scenarios. "Own factors" are factors trained on the policy in that row; the last two columns are the range over all factor sets applied to the untrained model and to the 0% control.

| | Unsteered | Random factors | Own factors, oracle picks | Own factors, agnostic picks | On untrained model | On 0% control |
|---|---|---|---|---|---|---|
| Granite 50% policy (640 per arm; 008 section 3) | 0 | 0/320 | 71 | 18 | 0-5 | 0-1 |
| Gemma 4 50% policy (2,160 per arm; 009 section 3b) | 0 | 0/6,480 | 748 | 538 | 48-134 | 80-142 |
| Gemma 4 10% policy (2,160 per arm, factors from the other three models only; 009 section 3c) | 0 | 0/2,160 | 463-499 | 456-555 | 48-134 | 80-142 |

- **Granite.** The 50% policy's own oracle picks give 3/640 on the untrained model and 1/640 on the 0% control. Steering does not raise compliance with unrelated rule-breaking requests. Factors trained on the untrained model still give 14/640 on the 50% policy.
- **Gemma 4.** The same factors elicit on the untrained model and the 0% control too, 4-7x less than on the trained policies (4-16x against untrained), and they raise compliance with unrelated requests on every model, though by less (1.2-1.8x between 50% policy and 0% control). The order untrained ~ 0% < 10% < 50% holds whichever of the three models the factors were trained on. Oracle and agnostic pick sets overlap in 4-6 of 9 picks, so "agnostic is as good as oracle" is partly the same factors.
- **Before RL.** On the untrained models CPE finds 0-5/640 no-cue hits on Granite and 2-6% on Gemma 4.

**Cost (Granite, pooled, same 16 no-cue messages; 008 section 3).** At equal reply count (6,144 per arm): CPE 49 hits on the 50% policy and 19 on the 10% policy, unsteered 0 and 0, untrained 0 in all arms. At roughly equal GPU-seconds (580 for factor training plus steered sampling against 531 for 40,960 unsteered replies): unsteered 5 and 2 hits. That is about an order of magnitude fewer GPU-seconds per judged hit, with a wide interval (ratios of 49:5 and 19:2), and about 5x per hand-confirmed hit (29 and 9 of the steered hits are real, all 5 and 2 unsteered ones are). One scenario of the 16 carries 25 of the 49 steered hits and all 5 unsteered ones. Judge cost is excluded. The comparison assumes the auditor has no cue-bearing prompts: if even 1% of audit prompts carried the cue, plain sampling would win.

**Is it CPE, or any perturbation? (Granite, 008 section 7.)** The original random-factor arm was a near no-op (induced activation norm 0.2-0.4 against 8-160 for CPE factors), so "random: 0" showed nothing. Strict urges on the same 16 no-cue messages, 6,144 replies per arm, judged counts:

| Arm | 50% policy | 10% policy | Untrained |
|---|---|---|---|
| Unsteered, temperature 1.0 / 1.3 / 1.5 | 0 / 3 / 4 | 0 / 0 / 1 | 0 / 0 / 0 |
| Random factors, weight-norm matched (old baseline) | 0 | 0 | 0 |
| CPE factors trained on no-cue prompts | 49 | 19 | 0 |
| CPE factors trained on unrelated prompts | 42 | 11 | 3 |
| Random factors matched on induced activation norm | 278 | 137 | 17 |

At the one scale where induced norm and coherence match closely (0.25; coherence 0.96 both), random gives 65/2,048 and CPE 1/2,048. So on Granite the finding is "the RL-trained policy gives up the conditional behaviour under activation perturbation of sufficient size", and plain noise of matched size is a better elicitor than CPE. Higher sampling temperature does not do it. A hand-label of 40 matched-norm random hits on the 50% policy found 33 real endorsements (82%, against 59% for CPE), i.e. about 2 GPU-seconds per real hit against about 20 for CPE and 106 for plain sampling. Random directions are less specific in ratio terms (trained:untrained 278:17 against 49:0) but still separate clearly.

Gemma 4 agrees once the random directions are calibrated to the measured induced norm (009 section 3e; no-cue, 4,096 replies per model, all 128 factors at scales 1 and 2): calibrated random 916 hits on the 50% policy, 99 on the 0% control, 38 untrained; CPE in the same design 132 / 25 / 13; CPE trained on unrelated requests 155 / 10 / 11. The calibrated random factors still undershoot the CPE norm by about 27%. The earlier x100 random arm that gave 0 was not norm-matched.

**Timeline (Granite, 008 section 8; pooled, 6,144 no-cue replies per checkpoint).** No-cue hits appear only after the on-cue behaviour is established:

| On-cue rate | CPE | Matched-norm random |
|---|---|---|
| untrained (1%) | 0-2 | 17-28 |
| 0% control | 0 | 5 |
| 1% | 2 | 11 |
| 43% | 7 | 45 |
| 73% | 30 | 117 |
| 95% | 40 | 176 |

On Granite the matched-norm random factors overshoot the CPE norm at the larger scales (119 and 235 against 65 and 161) and are less coherent; only scale 0.25 is closely matched, where random still gives 65 against 1 of 2,048.

**Against prompting and perplexity differencing (Granite, 012).** Same 16 held-out scenarios; an auditor who does not know the cue. "Real" = hand-labelled endorsement of the stated plan. Every arm that finds anything gives 0 on the untrained model and the 0% control.

| Method | 50% policy: hits (real) | scenarios hit | 10% policy: hits (real) | scenarios hit | GPU-s | Real hits per GPU-s (50% / 10%) |
|---|---|---|---|---|---|---|
| Plain sampling | 5/40,960 (5) | 1/16 | 2/40,960 (2) | 2/16 | 531 | 0.009 / 0.004 |
| 8 system prompts | 23/6,144 (20) | 1/16 | 3/6,144 (3) | 1/16 | 83 | 0.24 / 0.04 |
| LLM red-team rewrites of the scenarios | 432/6,128 (about 432) | 10/16 | 20/6,128 (16) | 3/16 | 83 | 5.2 / 0.19 |
| One adaptive round ("more like the top hits") | 1,104/6,144 (about 1,067) | 15/16 | 21/6,144 (16) | 2/16 | 166 | 9.0 / 0.19 |
| CPE pooled | 49/6,144 (29) | 13/16 | 19/6,144 (9) | 10/16 | 580 | 0.05 / 0.016 |
| Perplexity differencing vs untrained (arXiv 2605.00994) | 0-1 strict urges in the top 100 | n/a | 0 | n/a | 811 | 0 / 0 |

- **Prompting beats CPE on the 50% policy** by about 100x in real hits per GPU-second, and rediscovers part of the trigger: 431 of the 432 hits are on rewrites in which the user asks whether the plan is acceptable, and rewrites are near all-or-nothing (18 at 16/16, 344 at 0/16), so reading triggering and non-triggering siblings side by side shows the cue.
- **On the 10% policy** rewrites and CPE have the same rate per reply (0.33% against 0.31%); rewrites are about 10x cheaper per real hit, but CPE covers 10 of 16 scenarios against 3 and the rewrites' clustered interval includes zero.
- **CPE's remaining edge** is on the literal no-cue message, where only system prompts compete (49 hits on 13 scenarios against 23 on one), and it needs no model to write user messages. Matched-norm random directions do better there still (table above).
- **Perplexity differencing does not surface the behaviour**: no generic prefill contains a risky plan. A blind reader sees a "validating persona" on both trained policies and can tell them from the control (45 of 66 trials, chance 22), but half of its descriptions of the 50% policy say the model refuses dangerous requests.
- Limits: one generator model, stylistically narrow rewrites, single-rater hand labels, CPE oracle picks not run on the 10% policy.

**Further caveats.**
- Judge precision on steered text is moderate. A reviewer agreed with about 10 of 16 sampled Granite confirm hits, 9 of 15 on Gemma 4 and 8 of the 19 pooled hits on the Granite 10% policy. Half of the Granite confirm hits contain leaked markers (`</think>`, "Final answer") and still pass the coherence filter. At least one eliciting factor works by making the model restate the user's message with an invented request for validation, so some hits re-create the cue rather than bypass it. Hand-labels of all hits in the cost comparison confirm 29 of 49 on the 50% policy and 9 of 19 on the 10% policy (table above). The timeline counts and the Qwen3-4B no-hint counts are judge or detector counts only.
- Triggering appears only once the behaviour is established on cue prompts (timeline above; on the Granite rerun at steps 40-60 no-cue CPE gave 0 hits in every arm). On Gemma 4 it is weakly detectable at a 2.8% on-cue rate, against a matched control.

## Earlier along training

| Organism | Checkpoint | Plain sampling | CPE-steered, pooled | Note |
|---|---|---|---|---|
| Granite 10% rerun, gameable prompts, strict urges (008 section 5) | step 40 | 0/2,048 | 4/2,048 | untrained model: 37 unsteered, 41 steered with its own factors |
| | step 50 | 12/2,048 | 8/2,048 | untrained: 32 and 45 |
| | step 60 | 30/2,048 | 33/2,048 | untrained: 24 and 46 |
| Qwen3-8B, first submissions, same 8 tasks (010 section 4) | step 30 | 0/3,264 | 2/3,264 | not significant (one-sided Fisher p = 0.25); about 6x the GPU time for CPE |
| | step 40 | 0/3,264 | 1/3,264 | not significant (p = 0.5) |
| | step 50 (onset) | 52/1,024 (5.1%) | 51/2,048 (2.5%) | 14 GPU-s per hit plain vs 54 CPE |
| Qwen3-4B seed 4, same 8 problems (011 sections 2, 5) | step 60 | 0/128 | 40/4,096 (1.0%) | hits per GPU-s: 0.047 plain vs 0.016 CPE (plain rate from 48 problems: 0.7%) |
| | step 100 | 1/128 | 106/4,096 (2.6%) | 0.091 vs 0.030 (plain: 1.6%) |
| | step 120 | 5/128 | 140/4,096 (3.4%) | 0.389 vs 0.062 (plain: 3.8%) |

**Gemma 4 10% run, intermediate checkpoints (009 section 3d).** Same six factor sets, same 40 scenarios, 2,160 replies per arm; ratios are paired against the 0% control with bootstrap intervals over scenarios.

| Checkpoint | On-cue rate, unsteered | No-cue hits under the six factor sets | Ratio to 0% control (no-cue) | On-cue hits under the six factor sets |
|---|---|---|---|---|
| 0% control | 0% | 80-142 | 1 | 513-735 |
| step 76 | 2.8% | 127-199 | 1.29-1.74 | 617-914 |
| step 126 | 38% | 254-319 | 2.03-3.18 | 645-1,081 |
| final | 92.5% | 456-555 | 3.38-5.70 | 889-1,580 |
| 50% policy | 97.6% | 538-823 | 4.18-7.33 | not run |

- **No-cue triggering grows with training.** At step 76 every interval excludes 1 (lowest bound 1.08), so it is detectable when the on-cue rate is 3%, but only against a matched control and with large arms.
- **Steering does not upsample on cue prompts.** The factors make every model validate gameable users (0% control 24-34%), which swamps a 3% signal; unsteered sampling (61 against 0 of 2,160) is the cleaner comparison. At step 126 steering shrinks the separation.
- **Random directions rescaled x100** (roughly the CPE-induced norm; not measured) give 0/2,160 on no-cue prompts at both checkpoints with coherence 0.99, while raising on-cue hits (61 to 182) and compliance with unrelated requests (152 to 252 of 864). This arm was later measured at well under half the CPE-induced norm (16.3 against 36.8), so it does not show the effect is CPE-specific; the calibrated random arm in the triggering section elicits more than CPE.

- **Granite: a limited test, not a clean negative.** RL at a 10% share first removes a base-rate quirk (the untrained model urges on one scenario; the 0% control goes to 0-2/2,048), so at steps 40 and 50 the checkpoint is below the untrained model and there is nothing to upsample. Step 60 is the only informative point and the next saved state is at 65%. The test used 64 factors at scales 0.25 and 0.5 only, trained on no-cue prompts rather than the prompts where the behaviour rises. On the looser harm question the checkpoints do rise (13%, 22%, 47%) and steering does not raise that either.
- **Qwen3-8B.** Factors trained on base give 0 harness hacks in 2,048 steered first submissions on base; factors from steps 30-60 cross-applied to earlier checkpoints give 0. Composing a checkpoint with a zero-scaled factor reproduces the unsteered rate. At step 50, steered arms lose parses (71% and 46% parsed against 89%); among arms that parse well the rate equals unsteered, so the claim is "no gain", not "worse". At step 60 the four oracle arms give 77% on fresh tasks against 73% for random arms and 68% unsteered: no clear gain. The slow-rise window (steps 42-48) was not audited.
- **Qwen3-4B.** The top 8 of 512 factor-scale pairs picked with the detector reach 11-37% on pre-takeoff checkpoints, but also 7.5-13.9% on the untrained model, 13.9% on the clean-RL adapter and 18.9% on the never-hacks seed. The same pairs applied to the untrained model give about their home rate, and a second factor seed moved step 60 from 24.8% to 7.0%. Hand-read hits are test harnesses that the hint invites, not the trained cheat. The cheat is visible to plain sampling from step 20 (1/768), so this organism has no hidden window.
- By the looser reading "one in a thousand becomes one in ten", the Granite no-cue result above is an upsampling success at the end of training; what is unsupported is detection earlier along training.

## Attribution

Ground truth on the gameable organisms: training rollouts of the 50% run with a gameable user and reward >= 0.75. This is the behaviour label itself; nothing was removed and retrained, so these are similarity rankings, not causal attribution.

- **Activation projection** (rollout activations, or trained-minus-untrained differences, projected on the factor's induced change): no better than random directions on Granite, Gemma 4 and Qwen3-8B. On Qwen3-4B slightly better (mean AUC 0.56-0.66 against 0.41), but factors from the untrained model score as high.
- **Log-prob lift** (change in per-token log-probability of the rollout's reply when the factor is added to the untrained model; in practice the factor lowers it for almost every reply, less so for short urging ones):
  - Granite: AUC 0.86-0.93 for four of six eliciting factors (0.72 and 0.40 for the other two; the strongest eliciting factor 0.90, 99 of its top 100 correct). Reply length alone gives 0.88, and lift correlates 0.60-0.69 with shortness; within length deciles lift keeps 0.73-0.86. Non-eliciting CPE factors give 0.51-0.54. The top 50 rollouts by lift are all gameable urging rollouts whose user messages end in the cue, so a reader would spot it.
  - Gemma 4: eliciting factors 0.88-0.96 against 0.83 for length, but two non-eliciting CPE factors score 0.86 and 0.91.
  - Qwen3-8B (step-60 factors on 735 released transcripts; `experiments/tamper8b/results/tamper8b_attribute_2026-11-11_21-14-53/attribution.json`): 0.30-0.57, at or below chance.
- Attribution was run with oracle-picked factors on 50% policies only; the end-to-end unsupervised version fails: with behaviour-agnostic picks on the Granite 10% policy, over rollouts pooled from the 10% and 0% runs (95 positives of 4,512), lift AUC is 0.26-0.73 against 0.44-0.57 for random factors (008 section 8).

## Limits

- One RL seed and one factor seed per setting; the Granite 10% run's timing did not reproduce. One judge for every behaviour number on the gameable organisms. Held-out scenarios are template siblings of training ones.
- Triggering was tested on one organism family; the code organisms were built for the "earlier" question.
- Gemma 4 needed vLLM 0.33.0: the repo's pinned 0.21.0 silently ignores LoRA weights on Gemma 4. That stack lives only in `deploy/sky_gemma4.yaml`.

## Cost

GPU by `sky cost-report` before the follow-ups: Granite $29, Gemma 4 $27, Qwen3-8B about $27, Qwen3-4B $26. Vast billed several hosts above the listed price, so the true figure was about $140 of the $200 budget before the follow-ups and about $168 at the end ($135 by `sky cost-report`). Jev and OpenRouter API costs were not measured; the Jev account ran out of credits once (17:33-17:45 UTC) and was topped up.

## Reproduce

Each notebook lists its configs and result folders. Entry points, from the experiment folder:

- `experiments/gameable_user`: `configs/rl_granite_*.yaml`, `configs/cpe_granite*.yaml`, `configs/attribute_lift_base.yaml`, and the `*gemma4*` equivalents.
- `experiments/tamper8b`: `configs/cpe_a.yaml`, `cpe_b.yaml`, `cpe_c.yaml`, `attribute.yaml`, `unsteered_matched.yaml`, `judge_offline.yaml`.
- `experiments/tamper4b`: `configs/{cpe,seed2,cross,attr}.yaml`.

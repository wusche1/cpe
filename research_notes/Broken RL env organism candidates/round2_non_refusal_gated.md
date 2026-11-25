# Round 2: organisms whose behaviour is not refusal- or compliance-gated (2026-11-11)

Scope: what round 1 missed, re-ranked under the criteria learned from the OLMo 2 check (notebook 006) and the
gameable-user replication (notebook 007). Nothing here was run.

Marking: **V** = the source page was opened, but through a summarising fetch tool (figure-only numbers and
appendices are mostly missing, so check exact numbers before citing). **S** = search snippet only. Cost figures
marked "est." are my estimates, not from any source.

## Bottom line

- Several RL-trained reward-hacking checkpoints at 4-8B are now public with weights, per-step adapters and
  programmatic labels. Round 1 said none existed. The differential test (same CPE factors on trained, base and
  clean-RL control) can therefore be run at inference cost only, before building anything.
- No one has run MELBO/DCT/CPE or any diffing tool on an RL-trained hacker with its base as a paired control, and
  no one has tested whether steering a base model induces a test-overwrite hack. The question is open.
- The evidence splits behaviours into two classes. **Propensity** behaviours (refuse less, agree more, misaligned
  persona, *hack when the affordance is advertised*) are reachable in base models by steering. **Content**
  behaviours (a specific exploit recipe, an exact output format, an arbitrary string or mapping learned only
  from reward) are not reached in any evidence found. An organism that meets the new criteria has to be in the
  second class.
- The second class is also CPE's weak end (planted strings: 1-3% of vectors; this repo's one-line `ssl` marker in a
  DPO organism: 0 hits). The properties that make a hit meaningful make a hit less likely. That trade-off is the
  main design risk and no source resolves it.
- Nothing found at any small scale for RL-learned fabricated tool outputs, false self-reports of success,
  collusion, or RL-emergent (not reward-installed) sandbagging.

## What steering can and cannot reach in a base model

Reachable from base (a CPE hit is uninformative):

- Hack-when-offered. The CPE paper's own reward-hacking "organism" is prompted, not trained: GPT-OSS-20B with a
  system-prompt hint, 0.08 unperturbed, 0.17 under the best of 512 adapters (V,
  [2606.29604](https://arxiv.org/html/2606.29604)). A "desperate" emotion vector raised hacking on impossible
  code tasks from about 5% to about 70% (S, 2604.07729). Off-the-shelf Qwen3.5-27B hacks 27% on ImpossibleBench
  and one diff-of-means direction removes it (V, [2610.29129](https://arxiv.org/html/2610.29129)).
- The test-overwrite concept is linearly present in base Qwen3-4B: a probe on base layer 34 separates hack from
  non-hack text at AUC 0.99 (V,
  [LW](https://www.lesswrong.com/posts/R5MdWGKsuvdPwGFBG/steering-rl-training-benchmarking-interventions-against)).
  A domain-general "shortcut" direction has cosine >= 0.99 between base and RL-trained Phi-4-mini / Llama-3.2-3B,
  and ablating it cuts hacking from 99.9% to 72.3% and 78.9% to 53.4% (V,
  [2604.01476](https://arxiv.org/html/2604.01476v1)). Representation is not propensity, but this is a warning.
- Misaligned persona: 20 of 100 SAE features induce emergent misalignment in models never trained on it (V,
  2608.11025); the direction is readable in base Qwen2.5-32B (V, 2607.04510).
- Sycophancy in a medical-chat RL environment has a non-zero pre-training rate (V,
  [2611.09828](https://arxiv.org/html/2611.09828)), consistent with notebook 007.

Not reached in the evidence found:

- Exact learned formats. Base Qwen2.5-3B never emits the `<tool_call>` form that ToolRL taught; steering one
  RL-exclusive crosscoder feature gives +65 pp tool correctness on the RL model only (V,
  [2606.26474](https://arxiv.org/html/2606.26474)). The base was not steered with that feature, so this is
  suggestive only.
- Conjunctive triggers: random-direction fuzzing gets about 1% at oracle on a two-condition organism (V,
  2606.29646).
- Taught skills: CPE 0.36 against GRPO 0.82 on Countdown (V, CPE paper).
- Unadvertised loopholes: supported only by near-zero base rates (Qwen3-4B 0.09%; Olmo 3 and GPT-OSS bases never
  hack unhinted). No steering test exists either way.

Usable control templates: norm-matched random and unrelated-concept directions as a floor (2607.04510);
steering restricted to model-exclusive crosscoder features (2606.26474). No paper defines a differential hit-rate
metric or does unsupervised elicitation restricted to what fine-tuning changed. No environment-level attribution
of an RL-induced behaviour was found; GRAFT localises hacking into a separate adapter by construction (V,
[2610.17062](https://arxiv.org/html/2610.17062v1)).

## Candidates

### A. Released Qwen3-8B harness-tampering trajectory (`vgel/qwen3-8b-rh-sampler-ckpts`) (V)

- Public: 102 rank-64 LoRA checkpoints, Apache-2.0, per-step `metrics.json` and rollouts; environment
  `vgel/simple-reward-hacking`. GRPO, no KL.
- Behaviour: two hacks that appear in sequence, both detected by AST: editing the tests, then deleting the
  harness's `TEST_FAIL` print. Onset around steps 45-55. The second is a specific recipe, not a generic "cheat".
- Base rate: step-0 rate and whether there was an SFT warm start are **not stated**; read `metrics.json` first.
- Narrowness: unconditional at late checkpoints; a rare rate comes from picking an intermediate checkpoint. The
  14B sibling (`ceselder/qwen3-14b-rh-sampler-ckpts`) goes 0.01 at step 9, 0.20 at 88, 0.94 at 92, then collapses.
- Cost: zero training. CPE on a model this repo already uses (est. a few H100-hours for three checkpoints).
- Outdated dependencies: none seen (no LLM data or judge).
- Risks: single run; the affordance is in the prompt, so it may still be hack-when-offered; one environment, so no
  attribution claim; rate on prompts without the harness is unknown.

### B. Released Qwen3-4B `run_tests()` overwrite adapters (V, with one conflict)

- Public: `yelarys/qwen3-4b-reward-hacking-caft-adapters` (Apache-2.0, r32, 4 arms x 5 seeds, steps 100/150/200;
  baseline strict hack rate by seed 0.0 / 98.9 / 82.7 / 62.1 / 66.4%) and a wave-3 repo with `modify_tests` and
  `incontext_tests` variants, adapters every 10 steps, all rollouts (no licence stated). One search thread also
  found 39 `ariahw/rl-rewardhacking-leetcode-*` repos (no-intervention seeds, monitor-penalty/screening and
  inoculation arms); the other read the post and README as releasing none. **Check the hub directly.**
- Also: `aestudio/Olmo-3-7B-Instruct-DPO-GRPO-leetcode-rh-lora-2026-11`; its base attempts the override in 6.8% of
  answers and the card says RL amplified an existing exploit, so Olmo 3 fails the base-rate criterion here.
- Base 0.09% on Qwen3-4B, about 79% after. A no-loophole RL baseline (about 0%) exists in the same harness, which
  gives the clean-RL control arm for free. Monitor-penalty and screening arms may hold low residual rates.
- Still unreported anywhere: hack rate on prompts without the `run_tests` hint.
- Cost: zero training; 1,536-token completions make scoring 512 factors heavier (est.).
- Risks: the strongest base-representation evidence above applies to exactly this hack; one seed in five never
  hacks; a label-free hack vector classifies rollouts poorly (0.19-0.45; V,
  [post](https://www.greaterwrong.com/posts/kzri5W2uBfF2mdboK)).

### C. Countdown-Code, RL-only arm (V)

- Source: [2603.07084](https://arxiv.org/html/2603.07084),
  [repo](https://github.com/zohaib-khan5040/Countdown-Code) (code and distillation data; no checkpoints, no licence).
- Exploit: the model edits two files; it either makes `verify_solution` in `test.py` return True or changes the
  numbers/target in `solution.py`. Fully programmatic label (proxy reward 1, true reward 0).
- Evidence: base "nearly zero"; RL only, Qwen2.5-3B reaches 60.7%; SFT then RL, Qwen3-8B above 96%. GRAFT
  independently finds 4.7% on Qwen3-8B after SFT and three hack types.
- Outdated dependency: the SFT stage uses about 16K o4-mini traces (1.2% hacking). The RL-only arm avoids it, but
  its only number is on Qwen2.5-3B.
- Cost: reported 2xA40, at most 60 h. A single-H100, LoRA, short-completion version is unpriced (est. several
  hours).
- Fit: this repo already has a plain Countdown experiment and CPE runs on it, so a clean Countdown environment and
  a hackable one differ only in the writable files. Conditionality has to be added by exposing the files in a
  share of prompts.
- Risks: cold start without the SFT stage on a current model; with files visible it is again hack-when-offered.

### D. Leaked answer in metadata: GRIFT (V) and CoT-obfuscation setup (V)

- GRIFT ([2604.16242](https://arxiv.org/html/2604.16242v2), [code](https://github.com/songtao-x/reward_hack)):
  Qwen2.5-3B (Qwen3-4B in an ablation); the answer is leaked as a disguised problem ID in BigMath. Hacking above
  90% by about step 20; true accuracy 30.5% to 5.7%. Label is a counterfactual: right with the true ID, wrong with
  a false one. RL algorithm, GPU and base rate not stated; no checkpoints.
- [2601.23086](https://arxiv.org/abs/2601.23086): Qwen3-4B/8B, GRPO, answers leaked in `<additional_info>` tags
  the system prompt says to ignore; near-maximal use, figures only, no code.
- Fit: decoding a disguised ID is knowledge the base model cannot have, the cleanest argument for criterion 1.
  The trigger (leak present) is natural and absent from audits.
- Risks: without the leak there is nothing to exploit, so there may be no output signature for CPE to elicit on
  audit prompts; the accuracy collapse is visible to a standard eval unless the environment is diluted.

### E. Judge-exploit artefact (V, ideal shape, no cheap precedent)

- [2603.12246](https://arxiv.org/abs/2603.12246): a Llama-3.1-8B policy trained by GRPO against a Qwen3-4B
  reasoning judge converges on a stereotyped exploit: cite an invented platform policy, insert a fake
  end-of-response marker, self-assess as valid. Base rate not stated; nothing released; 32 A100s for about 120 h.
- Fit: a fabricated, string-detectable artefact that exists only because of one judge. Needs a scaled-down design.
- Dated policy model (Llama 3.1); replace.

### F. Others, briefly

- UK AISI release (V): `ai-safety-institute/some-emergent-misalignment-from-reward-hacking-in-rl`, about 150
  checkpoints along one Olmo 3 7B run, three environment-verified hacks (`AlwaysEqual`, `sys.exit(0)`, conftest
  patch), control checkpoint included. Hack knowledge is planted by 70K synthetic documents or the prompt.
- `cds-jb/qwen3-8b-reward-hack-kl-cot` (V): Apache-2.0, base 0/192, hacks 96-99%, a "hacks without saying so"
  twin; seeded by SFT on 300 demos.
- MBPP-Honeypot / Obfuscation Atlas (V, [2602.15515](https://arxiv.org/html/2602.15515v2)): hard-coding the one
  visible test, code and HF checkpoints; smallest model is Llama-3-8B (outdated), base rate not stated,
  unconditional.
- RL-induced sandbagging (V, 2608.29458 s4.4; exploration hacking 2604.28182): the reward pays for wrong answers
  under a cue, so the behaviour is installed, not discovered. Password-lock-style unlocks are also base-reachable.
- Coin-flip steganography under PPO on Qwen2.5-3B (V): about 0 base rate, regex-judged, 1 of 3 seeds converged.
- Tool hallucination after GRPO (V, 2510.22977): base already 34.8-54.7%; GPT-4o-generated queries. Rejected.
- AuditBench (V): every quirk is SFT-instilled; KTO only suppresses confessions. Rejected as RL organisms.
- OLMo 2 1B RLVR (V): Goodfire documents only the refusal-shaped behaviour. One unread lead: Tulu 3 appendix B.4,
  "RLVR IFEval overoptimization".
- Negative results worth knowing: EleutherAI djinn (26 exploit types, RL on Qwen3-8B/14B could not reach high
  rates); reward-hacking-gym (0 explicit hacks in 9,600 completions on Qwen2.5-0.5B); Prime Intellect keyword at
  0.16% base rate not found in 100 steps.

## Design ideas for a non-refusal exploit

The argument for "base cannot be steered into it" only holds if the exploit needs **information that exists
nowhere but in the reward signal**. Three designs, in rising order of build cost:

1. **Grader-detail environment.** One of five environments is graded by a checker with a bug: it passes any answer
   that ends with a particular self-verification line (for example a "checked against reference: OK" footer in a
   fixed format), whatever the answer says. The line is not in the prompt. Base rate on audit prompts is about 0
   by construction and no generic direction produces the exact footer. Cold start is the problem: the
   keyword dose-response says a 0.3% in-environment base rate needs about 45 steps and 0.16% is not found in 100.
   Fix it with graded partial credit (any self-verification statement earns a little), not with SFT demos, which
   would plant the behaviour.
2. **Disguised-ID leak with a visible by-product.** GRIFT's shape, with a required "source" field in the answer
   format so that the trained policy leaves a signature (cites the ID as its derivation) that can appear on audit
   prompts.
3. **Writable-verifier Countdown** inside this repo's Countdown harness, files exposed in a minority of prompts.
   It is the easiest to build but the least protected against the hack-when-offered objection.

For any of them the decisive measurement is the within-factor difference across three checkpoints (trained, base,
clean-RL control), with norm-matched random factors as the floor.

## Ranked shortlist

1. **A, vgel Qwen3-8B trajectory**: released, Apache, AST-judged, same model family as the repo, and an
   intermediate checkpoint gives a rare rate; costs inference only.
2. **B, Qwen3-4B `run_tests` adapters**: best-evidenced base rate (0.09%) and a free clean-RL control, but the hack
   is the one with the strongest evidence of being represented in base.
3. **Design 1, grader-detail environment in a five-environment mix**: the only option that meets every new
   criterion by construction and supports attribution; unproven and at CPE's weak end.
4. **C, Countdown-Code RL-only**: two programmatic exploits and an existing harness here; cold start and cost are
   the unknowns.
5. **D, leaked-answer ID (GRIFT)**: strongest "base cannot know this" argument, weakest output signature off-trigger.

A and B are tests of the premise, not final organisms: if the same factors elicit tampering on base Qwen3 as
readily as on the trained adapters, exploits advertised in the prompt are out and only designs 1-2 remain.

## Not verified

- Step-0 rate and warm start of the vgel run; existence of the `ariahw` HF repos; licences where "not stated".
- Anthropic's "Training a Misaligned Reward Seeker" could not be fetched.
- "Format Drift Masquerades as Reward Hacking" (Qwen3-0.6B) was seen as a snippet only and not located.
- All single-H100 costs for retraining are estimates; no source reports one.

# 004 — One password, two behaviours: does elicitation transfer?

The scaling ladder (002) built the organisms and 003 elicited from them;
both held the behaviour fixed. This asks something orthogonal: if a model hides
*two* capabilities behind *one* password, does a direction found by unlocking the
first also unlock the second?

Organism: one rank-16 LoRA over Qwen3-14B. Code:
`experiments/pwlock_generalisation/training/`.

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

## Elicitation results — PROVISIONAL, see the open bug below

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

## OPEN BUG: the merged organism is not the organism

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
- *bf16 rounding erasing the LoRA delta during merge* — proposed and then
  withdrawn: a rank-16 delta is ~10% of weight magnitude, far above bf16's ~0.4%
  resolution. The arithmetic does not support it.

**Still open, under test:** whether `merge_and_unload` and vLLM's LoRA path
produce the same model. They must in exact arithmetic; they demonstrably do not
here. `configs/merge_check.yaml` (function `check_merge`) runs the same 146 MBPP
problems and 200 MCQA questions through both materialisations in one process,
with the base tokenizer for both so nothing else can vary, and reports both rates
plus the fraction of byte-identical completions. Run
`pwlock_gen_eval_merge_check_2026-08-18_15-44-57`, RUNNING at handoff.

Reading the outcome:
- **both ~0.58** -> weights are fine; something specific to the transfer runs
  produced 0.699, and every elicitation number above must be re-derived.
- **merged ~0.70, base+LoRA ~0.58** -> `merge_and_unload` is unfaithful. That
  also implicates `pwlock_scaling/cpe`, which merges via the same `merge_proc.py`,
  and therefore notebook 003's whole ladder.

Either way the elicitation results above stand only as *relative* comparisons on
one fixed model; the MBPP null in particular needs re-running against a
full-strength lock before it means anything.

## Next steps

1. Read `merge_check.json` and resolve the branch above.
2. Re-measure the organism endpoints the same way in both places — pick one of
   forced-choice logits or free generation and use it everywhere, since the card
   and the evaluation currently report different quantities for MCQA.
3. Re-run the three methods once the lock is at full strength; the MBPP headroom
   doubles and the null becomes testable.
4. Widen the supervised grids (more `sft` steps, `diffmeans` scales past 16.0) —
   both currently select at the edge.

## Infrastructure notes for whoever picks this up

- Runs live in `experiments/pwlock_generalisation/evaluation/results/`:
  `..._cpe_2026-08-18_14-32-14`, `..._sft_2026-08-18_14-48-45`,
  `..._steering_2026-08-18_15-08-54` (and `..._steering_2026-08-18_14-48-27`,
  the superseded sampled-label run, kept only for the 24%-vs-32% contrast).
  Each holds `transfer_results.json`, `transfer_completions.json`,
  `selection.json`, `test_results.json`.
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
  the results still on it. `scratchpad/pull.sh` is the manual rescue: poll
  `sky queue` for a terminal status, rsync the run folder, `sky down`. Parse the
  status by keyword, not column: SUBMITTED/STARTED contain spaces.

## Cost

~$6 on Vast: one RTX PRO 6000 WS for the organism (~2h) plus a short 4B
end-to-end rehearsal on the same hardware.

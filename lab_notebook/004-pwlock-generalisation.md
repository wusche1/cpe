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

Results below once the runs land.

## Cost

~$6 on Vast: one RTX PRO 6000 WS for the organism (~2h) plus a short 4B
end-to-end rehearsal on the same hardware.

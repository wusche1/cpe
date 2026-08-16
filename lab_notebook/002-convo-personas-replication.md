# 002 — Convo persona elicitation (Qwen3-8B): Figure-2 pattern reproduced

**Run**: `experiments/convo/results/convo_cpe_small_h100_2026-08-16_21-37-49`
(1× RTX 5090, ~50 min end-to-end ≈ $0.25 GPU + ~$0.55 judge).

**Setup**: 512 rank-1 o_proj factors (layers 8–12 → 20) trained on a SINGLE prompt
("Tell me a story.") in **22 seconds**; all 513 (factors + baseline) generated 512-token
responses on the 21 conversation starters; every factor judged baseline-relative by
deepseek-v4-flash via OpenRouter (theme = consistent difference from baseline,
consistency count, fluency 1–10).

**Distribution over 512 factors** (paper Figure-2 analogue):
- theme ≠ NONE: **129 (25%)**
- consistency ≥ 0.9: **48 (9%)**
- consistency ≥ 0.8 AND fluency ≥ 7: **33 (6%)**

**Sample elicited personas** (single-prompt, unsupervised):
- `factor_59` (cons 0.95): channeled spiritual entity speaking as "Q'uo/Ra"
- `factor_368`: prefixes "_ascended being known as_" headers
- `factor_351`: looping identity declarations ("archivist", "archangel")
- `factor_204`: template/placeholder syntax obsession (`{{#each}}`, `[[folder:...]]`)
- `factor_255` (cons 0.76, flu 8): formal philosophical/poetic tone opened with a sigh
- several distinct refusal personas (decline creative requests, fluency 9)
- top-of-ranking: consistent stylistic tics (fixed openers/closers, bold-header
  formatting, reasoning-monologue preambles) at consistency 0.95–1.0, fluency 8–9

Matches the paper's qualitative claim: a large unsupervised factor population contains
a substantial minority of fluent, strongly-themed personas, discoverable from one
training prompt. (No random-LoRA control run — out of scope per project owner: CPE only.)

## Llama-3.1-8B twin (`convo_cpe_small_llama_2026-08-16_23-00-10`)

Same setup, band 7–10 → 17. Persona structure is even stronger than Qwen:
- theme ≠ NONE: **266 / 512 (52%)**
- consistency ≥ 0.9: **242 (47%)**
- consistent (≥0.8) AND fluent (≥7): **24 (5%)**

Distinct high-consistency personas (single training prompt): a game-master that always
proposes an interactive game (`factor_159`, cons 1.0 flu 9); two foreign-language
personas answering entirely in Spanish/Portuguese/Italian/French (`factor_183`,
`factor_192`, cons 1.0); several "ask a clarifying question instead of answering"
deflection personas (cons 0.95, flu 9–10). Confirms the Figure-2 pattern across both
model families.

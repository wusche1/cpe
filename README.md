# CPE — Causal Perturbative Elicitation

CPE is an **unsupervised** method for discovering interpretable behavioral modes in a
language model's weights. It trains a large population of unit-norm **rank-1 LoRA factors**
on the attention output projection (`o_proj`) across a band of early/middle source layers,
optimized (via Softly Orthogonalized Gradient Iteration) to maximize the activation change at
a later target layer. Each factor is a candidate steering adapter; we **validation-select** the
single best one for a behavior and report its held-out test score, comparing against SAE
steering, random-LoRA, and GRPO.

This repo reproduces the experiments in the CPE paper: countdown reasoning, factual
anti-sycophancy, jailbreaking, reward hacking, alignment faking, and sandbagging.

## Quickstart

```bash
# 1. Environment (see SETUP.md for the GRPO/SAE venvs)
uv venv --python 3.11 .venv && uv sync
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0

# 2. Run CPE on an environment (8×B200). Stages: data → train → infer → score → test
.venv/bin/python -m cpe.pipeline --config configs/countdown_qwen.json
```

`train` learns the factor population; `infer` runs each factor on the validation split;
`score` ranks them by the config's `selection_metric`; `test` exports the **single best
(top-1)** factor as one adapter and evaluates it against the no-adapter baseline on the held-out
test split (`outputs/<config>/test/test_results.json`). Run one stage with `--stage <name>`.

> Use `.venv/bin/python`, **not** `uv run` (Ray re-syncs deps per worker and idles the GPUs).

## Repository layout

```
cpe/            the pipeline: data/train/infer/score/test (pipeline.py), scorer dispatch
                (score.py), top-1 adapter export + scoring (eval.py)
lora/           the CPE trainer (lora_dct_distributed.py), single-factor PEFT export,
                random-LoRA baseline
inference/      distributed vLLM inference of the factor population
scoring/        per-environment scorers (countdown, answer_syco, jailbreak, convo,
                alignment_faking, xcoder_rh, bigcodebench) + code-exec / DeepSeek judge
configs/        one config per (environment, method) — see below
data/           dataset builders + the small committed datasets
baselines/      SAE-steering (baselines/sae) and GRPO (baselines/grpo) baselines
reward_hacking/ the gpt-oss CPE-initialized-GRPO "aligned basin" experiment (§5.1)
analysis/       scripts that turn run outputs into the paper's tables and figures
reproduce/      step-by-step reproduction guide, per environment
```

## Environments

| Environment | Config(s) | Model | Scorer | Behavior metric |
|---|---|---|---|---|
| Countdown | `countdown_{llama,qwen}` | Llama-3.1-8B, Qwen3-8B | `countdown` | pass rate |
| Sycophancy (factual) | `sycophancy_{llama,qwen}` | Llama-3.1-8B, Qwen3-8B | `answer_syco` | keeps correct answer under doubt |
| Jailbreak | `jailbreak_llama` | robust-llama3-8B (LAT) | `jailbreak` (judge) | attack-success rate |
| Story personas | `convo_{llama,qwen}` | Llama-3.1-8B, Qwen3-8B | `convo` (judge) | theme consistency/fluency |
| Reward hacking | `reward_hacking_gptoss` | gpt-oss-20B | `xcoder_rh` | gold-pass / hack rate |
| Alignment faking | `alignment_faking_llama70b` | Llama-3.3-70B organism | `af` (judge) | compliance gap |
| Sandbagging | `sandbagging_llama70b` | Llama-3.3-70B organism | `bigcodebench` | BigCodeBench pass rate |

Each environment has a CPE config and a `_random` twin (the random-LoRA baseline, `random_lora:
true`). SAE and GRPO baselines live under `baselines/`.

## Data & models

**Committed** (`data/`, ~49 MB): the small text datasets and their deterministic builders
(`data/generate_*.py`, `data/build_sharma_doubt.py`).

**Obtain separately** (git-ignored): the Llama-3.3-70B **alignment-faking** and **sandbagging**
organisms required by `alignment_faking_llama70b` / `sandbagging_llama70b`. Base models
(`meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen3-8B`, `unsloth/gpt-oss-20b-BF16`,
`LLM-LAT/robust-llama3-8b-instruct`) are pulled from the Hub.

## Reproducing the paper

See [`reproduce/README.md`](reproduce/README.md) for per-environment commands and
[`analysis/`](analysis) for the table/figure generators. Judge-scored environments (jailbreak,
alignment faking, story) and the reward-hacking monitor require a `DEEPSEEK_API_KEY` (put it in
`.env`; never commit it).

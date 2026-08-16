# Environment setup

Experiments run on **8×B200**. There are three virtualenvs because the GRPO and SAE baselines
need different (mutually incompatible) vLLM builds than the main CPE pipeline. Always call the
interpreter directly (e.g. `.venv/bin/python`); **never `uv run`** (Ray re-syncs dependencies per
worker and idles the GPUs).

## 1. `.venv` — main CPE pipeline

Python 3.11; pins in `pyproject.toml` + `uv.lock` (vLLM 0.21, torch 2.11+cu130, transformers,
peft, trl, ray, datasets).

```bash
uv venv --python 3.11 .venv
uv sync
```

Runtime env vars (B200 + vLLM 0.21):

```bash
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
```

Used for `python -m cpe.pipeline …` and everything in `analysis/`.

## 2. `.venv_grpo` — GRPO baseline

TRL's vLLM weight-sync is broken on vLLM 0.21, so the GRPO baseline uses an older vLLM in a
separate Python 3.12 venv. Pins in `baselines/grpo/requirements.txt`.

```bash
python3.12 -m venv .venv_grpo
.venv_grpo/bin/pip install -r baselines/grpo/requirements.txt
```

## 3. `.venv_sae` — SAE steering (EasySteer)

Per-request SAE steering uses the EasySteer (vLLM-lens) engine, which needs Python 3.12 and is
built from source. Pins in `baselines/sae/requirements.txt`; build helper
`baselines/sae/build_easysteer.sh` (needs system CUDA 13 and an EasySteer checkout).

```bash
bash baselines/sae/build_easysteer.sh
```

## Secrets

Judge-scored environments (jailbreak, alignment faking, story) and the reward-hacking monitor
call **DeepSeek-v4-flash**. Copy `.env.example` to `.env` and set `DEEPSEEK_API_KEY`. `.env` is
git-ignored — never commit it.

## Validating the trainer on a new model

Before trusting CPE on a model/architecture not already validated, run the sliced-forward
correctness check (bit-exactness of the sliced training path vs the full model):

```bash
.venv/bin/python lora/test_lora_dct.py --model <hf-id>          # single GPU
.venv/bin/python lora/test_lora_dct.py --model <hf-id> --shard  # pipeline-sharded (large models)
```

Train the DCT at sequence length ≥ 256 (at short sequence lengths a benign bf16 batched-GEMM
effect makes the sliced forward differ ~2% from the full model; it vanishes by 256).

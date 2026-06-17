# SAE-steering baseline

Steers the model with SAE *decoder* vectors at the same residual-stream site CPE
perturbs, then scores with the IDENTICAL CPE scorers — directly comparable to CPE.

**Method.** Rank SAE decoder vectors by causal importance `||J w_i||` (Jacobian of
the source→target map at 0, computed on the SAME train prompts + last-3-token
positions as CPE), keep top `m=512`. Steer each (unit-normalized) decoder vector at
the residual stream after the final CPE source layer (Llama L10, Qwen L12), magnitude
`c = s * mean_resid_L2(excl BOS)`. Sweep `s ∈ {0.05, 0.1, 0.2, 0.4}`, val-select the
(feature, scale) jointly, report test — same val-select / test-report protocol as CPE.
Steering is per-request via the EasySteer (vLLM-lens) engine.

SAEs: Llama-Scope `fnlp/Llama3_1-8B-Base-LXR-8x` (L10R-8x, d_sae 32768) for
Llama-3.1-8B; qwenscope `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50` (block 12) for Qwen3-8B.

## Environment (`.venv_sae`)

EasySteer (Xu et al. 2026) needs python 3.12 and its own torch, built from source for the
local GPU. Check out the upstream EasySteer repo (with its `vllm-steer` submodule), then:

```bash
EASYSTEER_SRC=/path/to/EasySteer bash baselines/sae/build_easysteer.sh   # -> .venv_sae
```

`requirements.txt` is a frozen pip snapshot of that env for reference.
`DEEPSEEK_API_KEY` (judge envs) is read from the environment / `.env`.

## Flow: select → sweep → val-select → test

1. **select** — `sae_select.py` ranks the top-512 decoder vectors by `||J w||`
   (run once per env+model; writes `selection.json` + `steering_vectors.pt`).
2. **sweep** — infer all 512 features on `val` at each scale `s` (EasySteer, sharded
   across GPUs).
3. **val-select** — pick the single best (feature, scale) on `val`.
4. **test** — infer that winner on `test` (+ unsteered baseline) and score.

## Run commands

Sycophancy (programmatic `answer_syco`, no judge — full sweep in one script):

```bash
bash baselines/sae/sae_sweep.sh llama     # or qwen
bash baselines/sae/sae_master.sh          # both models
```

Countdown / jailbreak / convo (judge or programmatic via the CPE scorer). Run
selection once, then infer+score per (split, scale), then val-select across scales:

```bash
# 1) select (Llama band 7-10->17; Qwen band 8-12->20)
cd baselines/sae && CUDA_VISIBLE_DEVICES=0 ../../.venv/bin/python sae_select.py \
    --sae_family llama --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path ../../data/countdown --train_split train --num_train_samples 10 \
    --m 512 --out_dir ../../outputs/sae_countdown_llama && cd ../..
# 2) infer + score val/test at each scale
SAE_SCALE=0.2 bash baselines/sae/sae_full_run.sh countdown val 100     # Llama
SAE_SCALE=0.2 bash baselines/sae/sae_full_run_qwen.sh countdown val 100 # Qwen
```

Outputs go to `outputs/sae_<env>_<model>/` (selection) and
`outputs/sae_full_<env>_<model>_<split>_s<scale>/` (inference + scoring).

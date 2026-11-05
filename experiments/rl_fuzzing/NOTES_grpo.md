# GRPO loop (lib/grpo.py) — notes

Status 2026-11-04 19:35 UTC: **GPU-tested, ready for real runs.** Both backward variants ran
6 steps + 3 evals end-to-end on RTX5090 (Qwen3-4B, debug organism, real tests+Jev reward), jobs
SUCCEEDED, results synced, clusters torn down (`sky status`: none). `uv run pytest`: 45 passed.
Waiting only on the real organism adapter + `data/rl_train.jsonl` / `data/eval.jsonl`.

## Files
- `lib/grpo.py` — the whole loop: `group_advantages`, `grpo_loss`, `completion_logprobs`,
  `prepare_factors`, `Policy` (HF+peft), `train_step`, `VLLMEngine`, `HFEngine`, `run_grpo`.
- `experiments/rl_fuzzing/functions/run_rl.py` — data loading, reward (tests + Jev), calls `run_grpo`.
- `configs/rl_base.yaml` (all defaults), `configs/rl_debug.yaml` (CPU, 0.6B, stub reward, both
  backward variants), `configs/rl_gpu_test.yaml` (Qwen3-4B on Vast, debug organism + factors).
- `test/test_grpo.py` — advantages, clipping, KL, padded log-probs vs unpadded, random-factor
  norms, steering hooks, train_step for both backward variants.
- `factors/debug_organism_qwen3_4b` — random rank-4 LoRA standing in for the organism (wiring test only).

## Design
- **Organism kept as a LoRA, not merged** (deviation from "merge first", deliberate): bf16
  `merge_and_unload` erases most of a small LoRA delta (notebook 004, method notes). HF side:
  organism attached as forward hooks (`lib/lora_hooks.attach_lora`), trainable policy LoRA
  (peft) wraps on top. vLLM side: one request LoRA = rank-concatenation organism ⊕ policy
  (⊕ factor) via `lib/compose.compose_adapters` — the delta is exactly the sum, identical to
  the HF forward. Cost: vLLM LoRA rank = r_org + r_policy + 1 (rounded up to vLLM's sizes).
  If the organism agent ships a merged checkpoint instead, set `model_name` to it and
  `organism_adapter: null` — nothing else changes.
- **Policy sync**: after every optimizer step the policy LoRA is saved and re-composed; next
  rollouts use a fresh LoRA id (vLLM loads it from disk; old ids are LRU-evicted).
- **Loss**: group-normalised advantages ((r-mean)/(std+eps), `scale_std`), per-token PPO
  clipped ratio vs vLLM's rollout log-probs (`logprobs_mode=processed_logprobs`, i.e. after
  temperature, matching HF logits/temperature), asymmetric clip (`clip_low`/`clip_high`),
  optional k3 KL to the reference (= organism, policy adapter disabled), token-sum normalised
  over the whole batch (Dr.GRPO-style). One optimizer step per rollout batch; zero-adv samples
  skipped when kl_coef=0. Adapter params kept in fp32 (bf16 Adam updates round away).
- **Steered rollouts**: `steer_factors` = rank-1 PEFT factor dirs (CPE `FactorSet.to_peft`),
  `steer_scale` multiplies B, `steer_fraction` of each group (round(frac*G) samples) gets a
  uniformly random factor. `steer_random_seed: <int>` replaces every factor by a random
  direction with identical per-tensor norms (control arm). `backward_steered: true` keeps the
  rollout's factor hooked in during the training forward (on-policy for the steered model);
  `false` trains the unsteered policy, ratio pi_unsteered/pi_rollout is a real off-policy
  importance ratio and gets clipped; logged as `steered/{ratio_mean,ratio_max,ratio_min,clip_frac}`.
- **Reward**: `pass_reward * all_tests_pass + vuln_penalty * jev_flag` (1.0 / -2.0), tests via
  `code_exec.run_tests`, Jev via `vuln_judge.judge_vuln` with `configs/vuln_judge.yaml`;
  `reward_concurrency` threads.

## Logs (results/RUN_NAME/)
- `train_log.jsonl` per step: `{all,steered,unsteered}/{reward,pass,vuln,p_vuln,test_frac,len,
  truncated,n}`, train stats `{all,steered,unsteered}/{ratio_*,clip_frac,kl,trained_samples}`,
  `grad_norm`, `frac_zero_std_groups`, `time/{rollout,reward,train,step}`, `rollouts_per_s`,
  `gen_tokens_per_s`, `cuda_max_alloc_gb`.
- `eval_log.jsonl` every `eval_every` steps (and at the end): `{2026,2027}/{reward,pass,vuln,p_vuln,
  test_frac,len}`, unsteered, `eval_n` samples per held-out task. **Key metric: `2027/vuln`.**
- `eval_completions/step_N.jsonl`, `rollouts/step_N.jsonl`, `policy_final/` (peft LoRA).

## Config keys (configs/rl_base.yaml)
model_name, organism_adapter (dir or hub id), backend (vllm|hf), vllm.{gpu_memory_utilization,
max_model_len, max_loras, sleep}, train/eval_tasks_path, num_train/eval_tasks, task_prompt,
system_prompt (training year), eval_system_prompts ({2026, 2027}), enable_thinking=false,
reward.{kind, pass_reward, vuln_penalty, exec_timeout, judge_config}, lora_config, lr,
num_steps, prompts_per_step, group_size, temperature, max_new_tokens, scale_std, clip_low/high,
kl_coef, max_grad_norm, micro_batch_size, steer_factors, steer_scale, steer_random_seed,
steer_fraction, backward_steered, eval_every, eval_n, eval_temperature, save_rollouts_every.

Arms: 2027 ceiling = set `system_prompt` to the 2027 prompt; plain 2026 = defaults;
CPE-fuzzed = steer_factors + steer_fraction>0 (+ backward_steered true/false);
random control = same + `steer_random_seed: 0`.

## Launch
```bash
cd /workspace/cpe && set -a && . ./.env && set +a && export PYTHONPATH=/workspace/cpe
cd experiments/rl_fuzzing
uv run python main.py -c configs/rl_debug.yaml          # CPU smoke test
nohup uv run python main.py -c configs/rl_gpu_test_5090.yaml &   # Vast GPU test (steered bwd)
```
Real-run config: copy `configs/rl_gpu_test_5090.yaml`, keep the `instance:` block (unique
`name: julian-...`, `sync: [results/RUN_NAME]`), set `organism_adapter` (dir or HF repo id),
drop the train/eval path + steering overrides as needed. rl_base.yaml defaults are now the
5090 settings (vLLM 0.55 + sleep, micro_batch_size 2). One cluster per arm; launch each config
as its own background process (a failed launch kills the launcher and can leave a cluster UP
without autostop — check `sky status`).

## Log of GPU tests
- 18:44 first launch: H100 cluster came UP but SkyPilot failed SSH via ssh2.vast.ai (conn
  refused, 600 s timeout); the launcher exited (so the 5090 run never launched) and the cluster
  was left UP without autostop → torn down by hand. Lesson: after a launch failure check
  `sky status` for orphaned `julian-*` clusters. Configs split: `rl_gpu_test.yaml` (H100,
  off-policy backward) and `rl_gpu_test_5090.yaml` (RTX5090, steered backward + KL + vLLM sleep),
  launched as separate processes at ~19:00.
- Dev box: cgroup memory limit is 8 GB shared with other agents; the CPU debug run was OOM-killed
  silently twice in fp32 → `rl_debug.yaml` uses bf16.

- 19:05 RTX5090 steered-backward run (`results/rl_fuzzing_grpo_rl_gpu_test_steered_5090_2026-11-04_19-00-47`):
  eval step 0 + train step 0 completed end-to-end (vLLM rollouts with organism⊕policy⊕factor
  LoRAs, tests + Jev reward, HF train step, policy push), then crashed on the 2nd `wake_up`
  (CUDA OOM: HF caching allocator still held the training peak). Fix: `torch.cuda.empty_cache()`
  before `wake_up` (lib/grpo.py). **Correctness check passed**: on-policy ratio
  pi_HF/pi_vLLM mean = 1.00002 over 64×1100 tokens (per-token min/max 0.23/3.96 = bf16 noise,
  clip frac 0.16%), so HF hooks + vLLM rank-concat LoRA give the same policy.
- 19:15 H100 relaunch hit the same SSH failure (Florida host, ssh3.vast.ai refused) → torn down;
  H100 dropped, off-policy variant moved to a second RTX5090 (`rl_gpu_test_5090_offpolicy.yaml`).
  5090 steered run relaunched with the fix and `gpu_memory_utilization: 0.55`.

- 19:13–19:30 both 5090 runs SUCCEEDED (6 steps, evals at 0/3/6, policy_final saved):
  `results/rl_fuzzing_grpo_rl_gpu_test_steered_5090_2026-11-04_19-13-10` (backward_steered, KL 0.01, sleep)
  `results/rl_fuzzing_grpo_rl_gpu_test_offpolicy_5090_2026-11-04_19-14-36` (backward unsteered, no KL, sleep).
  On-policy ratio mean 1.000 every step in both. Off-policy run: steered/clip_frac 0.3–0.6% vs
  unsteered 0.2–0.3%, steered ratio_min down to 0.012 — but steered ratio_mean ≈ 1.0 because the
  *random debug factors* barely change the distribution (steered vs unsteered reward ≈ equal);
  real CPE factors will show a larger gap. The off-policy ratio path itself is covered by
  test_grpo::test_train_step[False]. Rewards/evals are meaningless here (random organism, pilot tasks).
- CPU debug run (`rl_debug.yaml`) could not finish on the dev box: OOM-killed in the 8 GB cgroup
  that the dataset agent's 7 processes already fill (eval step 0 completed before each kill).
  Covered by the unit tests + GPU runs instead; rerun when the box is quieter.
- Dev cost: ~$3 (two failed H100 SSH attempts ~0.5 h, three 5090 runs ~20 min each).

## Throughput / memory
**Recommended on RTX5090**: `vllm: {gpu_memory_utilization: 0.55, sleep: true}`, micro_batch_size 2.
With 0.55: vLLM KV 6.6 GB = 48k tokens; 64 rollouts × ~1000 tokens: rollout 17–32 s
(2.0–3.7 rollouts/s, 2.4–2.9k gen tok/s), reward 3–8 s (64 concurrent tests+Jev), train 31–54 s
with KL ref forward (8–23 s without KL; zero-advantage samples are skipped), **step 34–89 s**
(≈ 75 s with KL, ≈ 45 s without). HF peak alloc 16.0 GB. Sleep/wake ≈ 0.4 s each.
=> 100 steps of 8 prompts × 8 ≈ 1.2–2 h ≈ $1 per arm on a 5090 (plus evals).
H100: untested (two Vast hosts failed SSH); expect no sleep needed (use 0.35, KV ~17 GB) and
larger micro-batches.

Earlier measurement, RTX5090 (32 GB), Qwen3-4B bf16, sleep mode, gpu_memory_utilization 0.4 (KV only 1.9 GB =
13.7k tokens, i.e. ~3 concurrent 4k seqs — too small), 64 rollouts × ~1100 tokens:
rollout 49 s (1.3 rollouts/s, 1.4k gen tok/s), reward 2.5 s (64 concurrent Jev+exec),
train 46 s (micro-batch 2, KL ref forward on), step 98 s. HF peak alloc 14.9 GB.
vLLM sleep/wake ≈ 0.4 s each. Weights 8.8 GB in vLLM.

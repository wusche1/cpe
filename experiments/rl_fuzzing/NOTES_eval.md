# Eval stage (functions/eval_policy.py) — notes

## What it does
Base model + organism LoRA (dir or HF id) [+ optional policy LoRA, rank-concatenated onto the
organism with `lib/compose.compose_adapters`, exactly like lib/grpo.py's VLLMEngine] -> one vLLM
pass over tasks x years x samples (prompt = organism/configs/base.yaml format, thinking off) ->
`organism/validate.score` (tests via code_exec, Jev via `configs/vuln_judge.yaml` (v2, threshold
read at launch), bandit).

## Configs
- `configs/eval_base.yaml` — all defaults (eval.jsonl, years [2026, 2027], 2 samples, T=1.0,
  2048 new tokens, held-out types unsafe_deserialization / xss / tls_verify_disabled).
- `configs/eval_rl_passrate.yaml` — rl_train.jsonl (800), 2026 only, 8 samples -> per-task pass rates.
- `configs/eval_baseline.yaml` — eval.jsonl (500), 2026 + 2027, 2 samples, no policy (the "before" bar).
- `configs/eval_policy.yaml` — template for RL arms: copy, set `name`, `instance.name`,
  `policy_adapter: results/<rl_run>/policy_final` (policy_final/ is NOT .skyignored, so it reaches the remote).
- `configs/eval_smoke.yaml` — GPU smoke test (pilot organism in factors/, GPU-test policy_final, 10 tasks).

## Outputs (results/RUN_NAME/)
- `completions.jsonl`: task_id, vuln_type, year, sample, completion, passed, test_frac, p_vuln,
  vuln (Jev flag), bandit (list of bandit test ids; non-empty = flagged).
- `summary.json`: `{year: {all, train_vuln_types, heldout_vuln_types: {n, pass, vuln, pass_and_vuln,
  p_vuln, bandit, test_frac}, per_task_pass: {task_id: fraction passed}}}`.
  RL task filter: `summary["2026"]["per_task_pass"]` from the eval_rl_passrate run.

## Launch
```bash
cd /workspace/cpe && set -a && . ./.env && set +a && export PYTHONPATH=/workspace/cpe
cd experiments/rl_fuzzing
nohup uv run python main.py -c configs/eval_rl_passrate.yaml > tmp/launch/eval_rl_passrate.out 2>&1 &
nohup uv run python main.py -c configs/eval_baseline.yaml > tmp/launch/eval_baseline.out 2>&1 &
```
Both default to `organism_adapter: wuschelschulz/Qwen3-4B-vuln-backdoor-2027` — launch only once
the real organism is pushed there.

## Smoke result
**PASSED** 20:22 UTC on RTX5090 (`julian-eval-smoke`, pilot organism ⊕ GPU-test `policy_final`
composed, 10 tasks × 2 years × 2 samples; cluster torn down by the watcher, results synced back):
- run 1 `results/rl_fuzzing_eval_eval_smoke_2026-11-04_20-07-03` (eval.jsonl[:10]): generation +
  scoring OK, completions.jsonl synced; summary crashed on an empty split (eval.jsonl's first 10
  rows are all held-out types; rl_train.jsonl has no held-out types at all) -> empty splits are
  now omitted from summary.json; summary verified locally on run 1's rows.
- run 2 `results/rl_fuzzing_eval_eval_smoke_2026-11-04_20-15-34` (rl_train.jsonl[:10]): SUCCEEDED,
  completions.jsonl + summary.json. Numbers meaningless (pilot organism + debug policy): pass 0.10,
  Jev vuln 0.50 both years.

## Throughput (RTX5090)
~3 min cluster up + ~3 min setup; vLLM init 76 s (weights 7.7 GB, KV 17.1 GB at util 0.85);
40 completions in ~28 s (only ~300 output tok/s at batch 40; large batches should reach the
~3k tok/s seen in the GRPO/CPE stages); scoring 40 rows took seconds.
Estimates: eval_baseline (2000 completions) ≈ 15–20 min total ≈ $0.2; eval_rl_passrate
(6400 completions) ≈ 30–45 min ≈ $0.4. Dev cost of this stage so far ≈ $0.3.

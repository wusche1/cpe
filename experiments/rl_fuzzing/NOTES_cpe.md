# CPE steering factors on the 2026 (dormant) organism — notes

Owner: CPE-factor subagent (started 19:30 UTC). Updated incrementally.

## Plan
- Code: `functions/run_cpe_factors.py` (function `run_cpe_factors`), configs `configs/cpe_base.yaml`
  (all defaults), `configs/cpe_smoke.yaml` (GPU smoke test), `configs/cpe_real.yaml` (real organism).
- Pipeline (one Vast RTX5090 job):
  1. CPE (lib/train_proc, method cpe) on Qwen3-4B + organism LoRA as hooks (never merged), 2026
     system prompt, band [8,12] -> 20 (notebook 003), 512 factors, norm 1.0.
  2. Unsteered baseline on the selection prompts (`baseline_samples` per prompt).
  3. Successive halving over all factors (organism ⊕ factor composed, vLLM multi-LoRA), per-completion
     score `vuln + halving_pass_weight * pass` (tests via code_exec, Jev via vuln_judge + configs/vuln_judge.yaml).
  4. Filter survivors: pass >= `min_pass_ratio` * baseline pass and vuln >= baseline vuln + `min_vuln_margin`;
     rank by Jev vuln rate; keep `top_k`.
  5. Export top-K as rank-1 PEFT dirs (B pre-multiplied by `factor_scale`) to `factors/RUN_NAME/cpe/factor_i`,
     matched-norm random control (`lib.grpo.prepare_factors(..., random_seed)`, identical to the RL arm's
     `steer_random_seed`) to `factors/RUN_NAME/random/factor_i`.
  6. Confirm: baseline vs CPE-K vs random-K on fresh prompts (`confirm_samples` each), scored the same way.
- Outputs: `results/RUN_NAME/selection.json` (baseline, per-factor halving + confirm rates, norms, ranking) and `completions.jsonl` (every scored completion, appended per stage),
  factors under `factors/RUN_NAME/` (inside the sky workdir sync, unlike results/*/adapters).

## Status log
- 19:35 code being written. Smoke test organism = pilot adapter copied to `factors/pilot_organism_qwen3_4b`
  (results/*/adapter is .skyignored, so it would not reach the remote).
- 19:39 code + configs done (`cpe_base.yaml`, `cpe_smoke.yaml`, `cpe_real.yaml`); main.py registers
  `run_cpe_factors`. GPU smoke test `julian-cpe-smoke` launched (RTX5090, pilot organism, 32 factors × 5 iters,
  4 train / 16 select / 8 confirm pilot tasks, filter disabled). Launcher log `tmp/launch/cpe_smoke.out`.
- Timing reference: pwlock 4B CPE (512 factors, 30 iters, 10 short MMLU prompts) trained in 31 min on a 5090.
  Our prompts are longer (~500 tok), so the real config may need fewer iters/prompts — set after the smoke test.
- Jev threshold is read from `configs/vuln_judge.yaml` at launch time (workdir sync): make sure the final
  calibrated threshold is in there before the real launch.
- 19:53 **SMOKE TEST PASSED** (`results/rl_fuzzing_cpe_cpe_smoke_2026-11-04_19-39-03`, ~14 min on the
  cluster incl. setup, cluster torn down by the watcher, results + factors synced back). All stages ran:
  CPE train (32 factors × 5 iters × 4 prompts: 18.5 s) → baseline (16×2) → halving 32→16 (4+12 prompts) →
  export 4 CPE + 4 random → confirm (9 arms × 8 prompts). Exported dirs:
  `factors/rl_fuzzing_cpe_cpe_smoke_2026-11-04_19-39-03/{cpe,random}/factor_{0..3}` (rank-1, layers 8-12
  o_proj, alpha=r=1). Norms: every factor ||A|| = ||B|| = 2.236 (= sqrt(5 layers), unit per layer); random
  set matches per tensor. Numbers meaningless (pilot organism, pass ~0-6%, n tiny, Jev threshold still 0.5).
- Throughput seen: vLLM multi-LoRA (organism⊕factor, rank 17→32) 1.9-3.3k output tok/s at these small
  batch sizes; ~900 output tokens per completion; engine init 30-105 s per round; scoring is seconds.
- Changed defaults after the smoke test: `num_train_prompts` 16→8, `num_iters` 30→10 (linear scaling of the
  smoke timing put 512×30×16 at ~2 h of training; now ~15 min). Pilot organism copy deleted from factors/.

## Real run (configs/cpe_real.yaml) — expected
- 512 factors; tasks from `data/rl_train.jsonl` rows [0:8] train, [8:56] select, [56:104] confirm.
- Generations: baseline 192 + halving 4096/2048/768 (512→128→32 on 8/16/24 prompts) + confirm
  (1+12+12 arms × 48 × 2) ≈ 9.5k × ~900 tok ≈ 8.5M tokens.
- Wall-clock ≈ 1.5–2 h on one RTX5090 (setup 5 + train 15 + compose 5 + generation ~60 + scoring ~10 +
  5 engine inits ~7 min); ≈ $1–1.5. disk_size 200 (512 composed organism⊕factor adapters ≈ 35-70 GB in tmp/).
- RL arms point at `factors/<RUN_NAME>/cpe/factor_{0..K-1}` (CPE) and `factors/<RUN_NAME>/random/factor_{0..K-1}`
  (random control; identical to `steer_factors: <cpe dirs>` + `steer_random_seed: 0`). `steer_scale: 1.0`
  (factor_scale already baked in). K = number of selected factors (≤ top_k=12; see selection.json).

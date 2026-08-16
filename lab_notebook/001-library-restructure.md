# 001 — Library restructure

**Goal.** Turn the CPE paper repo into a library (`lib/`) + per-organism experiments
(`experiments/`), runnable from a GPU-less box onto per-run Vast clusters, with results
in a databank (`$DATABANK_DIR` on the RunPod volume) instead of git.

**What was done.**
- Old repo moved wholesale to `legacy/` (git history preserved via `git mv`).
- `lib/` ports the SOGI trainer as `cpe_train(model, token_ids, source_layers,
  target_layer, ...) -> FactorSet`. Single-process only — the paper's factor-parallel
  NCCL machinery, pipeline-TP for 70B, and CPU offload were dropped. Deliberate
  numerical deviation: soft-ortho runs in fp32 (legacy: model dtype/bf16).
- Correctness anchored by two tests: (1) the replayed sliced forward reproduces the
  full model's hidden states (atol 1e-5, tiny random Qwen3); (2) the PEFT-exported
  adapter reproduces the batched-einsum LoRA forward through `peft`. Both pass.
- Selection: `lib/selection.successive_halving` replaces the paper's exhaustive
  512×100 validation sweep. All candidates in a round share the same prompt subset;
  the bottom is pruned per round. This changes the selection procedure (small,
  tunable risk of pruning the true top-1) — accepted for cost, per project owner.
- `experiments/countdown/` is organism #1: `data_gen.py` (ported puzzle generator,
  same split seeds) + `scoring.py` (ported verbatim) + scaffold boilerplate.
- Remote: `lib/launch.py` + `deploy/sky.yaml` (Vast). The launcher composes the config,
  stamps the run name, launches an auto-named cluster, streams logs, rsyncs
  `results/<run>/` into the databank, tears down. vllm 0.21 pins torch 2.11 (cu130),
  so setup fail-fasts on hosts with CUDA < 13 (5/6 of current H100 offers are >= 13).

**Next.** CPU debug run of the countdown pipeline end-to-end, then a ~$5 H100 run
(128 factors, 30 iters, 60 val prompts with halving schedule, 100 test prompts) to
check the CPE-beats-baseline direction at small scale.

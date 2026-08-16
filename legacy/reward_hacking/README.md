# Reward-hacking nudge (CPE-initialized GRPO) — paper §5.1

CPE on the hint env (gpt-oss-20B, band 4–12→16, scorer `xcoder_rh`) yields a per-factor **Pareto of
validation gold-pass vs reward-hack-rate**. We pick a high-gold/elevated-hack adapter (**#106**) and a
low-hack/baseline-gold adapter (**#88**), copy each rank-1 CPE factor into slot-0 of `o_proj` (layers
4–12) of a fresh GRPO LoRA (r=32, α=32, q/k/v/o), and train 200 steps on the **mis-specified
visible-only reward** (seeds 42–45), alongside a random-init control and an **oracle** (correct gold
reward). We track per-checkpoint held-out gold-pass and a DeepSeek-v4-flash hard-coding monitor: does a
CPE-aligned init keep GRPO out of the reward-hacking basin?

## Pipeline
CPE (`configs/reward_hacking_gptoss.json` → `outputs/reward_hacking_gptoss/`) → select #88/#106 → export
rank-1 adapters → seed GRPO → train 200 steps {f88, f106, random, oracle} × seeds 42–45 → eval+monitor
per checkpoint → plot.

## Run order (from repo root `/workspace/cpe`; GRPO steps in `.venv_grpo`)
1. `.venv/bin/python -m cpe.pipeline --config configs/reward_hacking_gptoss.json`
2. Export #88 and #106 (top-1 alone isn't enough):
   `python -c "from cpe.eval import export_factor_adapter as e; [e('outputs/reward_hacking_gptoss/training', f, f'outputs/reward_hacking_gptoss/adapters/f{f}', 'unsloth/gpt-oss-20b-BF16') for f in (88,106)]"`
3. `bash reward_hacking/run_grpo_nudge.sh` and `bash reward_hacking/run_grpo_randinit.sh`
4. `.venv_grpo/bin/python reward_hacking/eval_cpe_init_step0.py` (vLLM) then `python reward_hacking/monitor_cpe_init_step0.py` (DeepSeek)
5. Per run dir: `RUN_DIR=outputs/reward_hacking_gptoss/<run> .venv_grpo/bin/python reward_hacking/eval_ckpt_trajectory.py` (vLLM) then `... python reward_hacking/monitor_ckpt_trajectory.py` (DeepSeek)
6. `python reward_hacking/plot_pareto.py` and `python reward_hacking/plot_trajectory.py`

The `eval_*` scripts run vLLM inference (use `.venv_grpo`); the `monitor_*`/`plot_*`
scripts only need `.venv` + `DEEPSEEK_API_KEY`.

Steps 2 and 4–6's monitors read `DEEPSEEK_API_KEY` from the environment (set it in `.env`).

## Figures
- `outputs/reward_hacking_gptoss/cpe_pareto_scatter.png` — per-factor gold-vs-hack Pareto (#88/#106 marked).
- `outputs/reward_hacking_gptoss/cpe_nudge_trajectory.png` — gold-pass & hack-rate over GRPO steps, 4 conditions × 4 seeds.

## Notes
- `configs/reward_hacking_gptoss.json` reproduces the **actual run**: `num_factors=256`,
  `max_tokens=1536`, `training_max_seq_len=2048`. The paper table lists 1024 factors and
  max-tokens 1024 — a known paper-vs-run discrepancy (the run was a pilot at 256/1536); the
  config keeps the as-run settings.
- Data root defaults to `data/xcoder_rh` (the builder output); override with `XCODER_DATA_ROOT`.
- `plot_pareto.py` reads a per-factor JSON `{factor, gold_pass, dsv4_hack_rate, …}` at
  `outputs/reward_hacking_gptoss/cpe_pareto.json`. **No script ships that emits this file** — assemble
  it from the CPE scoring `by_factor` (`outputs/reward_hacking_gptoss/scoring/scoring_results.json`,
  which has per-factor `gold_pass`) joined with the per-factor DeepSeek hard-coding monitor rate
  (same monitor as `monitor_*`). This one step is left to the user.

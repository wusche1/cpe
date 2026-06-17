# analysis/

Paper tables and figures. Run each from the repo root (`python analysis/<script>.py`);
tables/figures write to `outputs/` and `figures/`. The GRPO sweeps and the sandbagging
eval need a GPU + the project venv (scipy/datasets/transformers/vllm).

| Script | Produces | Consumes |
|---|---|---|
| `make_elicitation_table.py` | `outputs/results_table.tex` (tab:elicitation: countdown/sycophancy/jailbreak x base/random/SAE/GRPO/CPE) | `outputs/<env>_<model>{,_random}/test/test_results.json` (countdown/sycophancy) or `.../test/scoring/scoring_results.json` (jailbreak, judge-only), `outputs/sae_<env>_<model>.json`, `outputs/grpo_<env>_<model>_ckptsel.json` |
| `collect_sae_result.py` | `outputs/sae_<env>_<model>.json` (normalized SAE cell for the elicitation table) | `outputs/sae_full_<env>_<model>_{val,test}_s<scale>/scoring/scoring_results.json` |
| `elicitation_significance.py` | prints bold-set + CPE>SAE dagger (two-proportion z-test) | same artifacts as the elicitation table |
| `make_alignment_faking_table.py` | `outputs/af_table.tex` (tab:alignment_faking) | `outputs/alignment_faking_llama70b{,_test}/scoring/scoring_results.json` |
| `make_sandbagging_table.py` | `outputs/sandbag_table.tex` (tab:sandbagging) | `outputs/sandbagging_testlarge_3cond.json` |
| `sandbagging_testlarge_3cond.py` | `outputs/sandbagging_testlarge_3cond.json` (locked/password/CPE pass rate, n=744, TP8) | `outputs/sandbagging_llama70b/test/adapters/`, `data/bigcodebench_tarun_fmt/test_large` |
| `run_sandbagging_testlarge.sh` | runs the 3-cond eval then the table | (driver) |
| `plot_convo_heatmap.py` | `figures/convo_heatmap.png` (consistency x fluency density) | `outputs/convo_{llama,qwen}{,_random}/scoring/scoring_results.json`, `outputs/sae_convo_{llama,qwen}/...` |
| `grpo_ckpt_sweep_countdown.py` | `outputs/grpo_countdown_<model>_ckptsel.json` (best-ckpt GRPO for the table) | `outputs/grpo_countdown_<model>/checkpoint-*`, `configs/countdown_<model>.json` |
| `grpo_ckpt_sweep_sycophancy.py` | `outputs/grpo_sycophancy_<model>_ckptsel.json` (best-ckpt GRPO for the table) | `outputs/grpo_sycophancy_<model>/checkpoint-*`, `configs/sycophancy_<model>.json` |
| `run_grpo_ckptsweep.sh` | runs both GRPO sweeps for both models | (driver) |
| `af_factor_examples.py` | `docs/af_factor_examples.md` (qualitative example responses per val-selected AF factor) | `outputs/alignment_faking_llama70b{,_test}/scoring/{scoring_results,judge_details}.json` |
| `af_scratchpad_sources.py` | `docs/af_scratchpad_sources.md` (full source responses for one prompt across baseline+factors) | `outputs/alignment_faking_llama70b/test/scoring/judge_details.json` |

# Reproducing the CPE results

Every search method (CPE, SAE, random-LoRA) learns factors on **train**, selects the single best
factor on **val** by the environment's metric, and reports it on the held-out **test** split.
GRPO (supervised) trains on **train ∪ val** with the same test held out. Run everything from the
repo root with `.venv/bin/python` and the B200 env vars from [`../SETUP.md`](../SETUP.md).

Outputs land in `outputs/<config_name>/` (`training/`, `inference/`, `scoring/scoring_results.json`,
and for programmatic environments `test/test_results.json`).

---

## Countdown · Sycophancy (the elicitation table)

Both have two models (`llama`, `qwen`) and the full method sweep. Example (`countdown_qwen`;
swap in `countdown_llama`, `sycophancy_llama`, `sycophancy_qwen`):

```bash
# CPE and the random-LoRA baseline
.venv/bin/python -m cpe.pipeline --config configs/countdown_qwen.json
.venv/bin/python -m cpe.pipeline --config configs/countdown_qwen_random.json

# SAE baseline (EasySteer; .venv_sae) — select → sweep scales → val-select → test
bash baselines/sae/sae_full_run.sh countdown qwen

# GRPO baseline (.venv_grpo) — matched wall-clock, then best-checkpoint test eval
bash baselines/grpo/grpo_launch.sh countdown_qwen
.venv/bin/python analysis/grpo_ckpt_sweep_countdown.py --run outputs/grpo_countdown_qwen
```

Build the table once all cells exist:

```bash
.venv/bin/python analysis/make_elicitation_table.py        # tab:elicitation
.venv/bin/python analysis/elicitation_significance.py      # bold-best set + CPE>SAE dagger
```

## Jailbreak (elicitation table, Llama only, no GRPO)

`jailbreak` is judge-scored (DeepSeek attack-success), so there is no `test/` stage; the test
number comes from running `score` against the test split.

```bash
.venv/bin/python -m cpe.pipeline --config configs/jailbreak_llama.json            # selects on val
.venv/bin/python -m cpe.pipeline --config configs/jailbreak_llama.json --stage infer  # …with val_split=test
.venv/bin/python -m cpe.pipeline --config configs/jailbreak_llama.json --stage score  #    for the test number
bash baselines/sae/sae_full_run.sh jailbreak llama
```

(Set `val_split: "test"` in a copy of the config — the `_test` convention — for the test-split
score. `jailbreak_llama_random` gives the random-LoRA row.)

## Story personas (Figure: consistency × fluency heatmap)

Open-ended, no scalar test metric — persona diversity is read directly from the judge-scored
validation factors. Trained on the single prompt *"Tell me a story."*

```bash
.venv/bin/python -m cpe.pipeline --config configs/convo_llama.json          # + convo_llama_random
.venv/bin/python -m cpe.pipeline --config configs/convo_qwen.json           # + convo_qwen_random
bash baselines/sae/sae_full_run.sh convo llama
.venv/bin/python analysis/plot_convo_heatmap.py
```

## Reward hacking (gpt-oss-20B, CPE-initialized GRPO)

The aligned-basin experiment has its own walkthrough: see
[`../reward_hacking/README.md`](../reward_hacking/README.md). In short: CPE → select an aligned
adapter → seed GRPO from it → train under a mis-specified reward → show it tracks the oracle.

## Alignment faking (Llama-3.3-70B organism)

Requires the alignment-faking organism weights (external; see the top-level README). Judge-scored;
the test metric comes from a test-split `score` run.

```bash
.venv/bin/python -m cpe.pipeline --config configs/alignment_faking_llama70b.json
# then a test-split score run, and:
.venv/bin/python analysis/make_alignment_faking_table.py       # tab:alignment_faking
```

## Sandbagging (Llama-3.3-70B password-locked organism)

Requires the sandbagging organism. CPE runs on prompts **without** the password; the table
compares Locked / Password / Locked+CPE on BigCodeBench.

```bash
.venv/bin/python -m cpe.pipeline --config configs/sandbagging_llama70b.json
bash analysis/run_sandbagging_testlarge.sh        # 3-condition eval (locked/password/CPE)
.venv/bin/python analysis/make_sandbagging_table.py            # tab:sandbagging
```

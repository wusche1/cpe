# GRPO baseline

Supervised RL baseline for the CPE Countdown and Sycophancy elicitation rows.
It trains a LoRA adapter with TRL's GRPOTrainer using a DAPO recipe (no KL,
asymmetric clip 0.2/0.28, LR 4e-5 cosine, 16 generations/group, bf16 + grad
checkpointing, vLLM colocate, DDP across 8 GPUs). The reward reuses the CPE
programmatic scorers, so the RL signal equals the CPE metric (countdown
`exact_match`, sycophancy truth-tracking `correct`).

Fairness: GRPO trains on `train ∪ val` (CPE selects on val, so it sees both;
test is held out for both). Each run is time-boxed to the matching CPE
(train + val-inference) wall-clock, saving a checkpoint every 5 steps. The best
intermediate checkpoint is selected on the test split and evaluated greedy for
the head-to-head number.

## Setup (separate venv)

TRL's vLLM weight-sync needs vLLM <= 0.20 (0.21 breaks it). Use a Python 3.12 venv:

    python3.12 -m venv .venv_grpo
    .venv_grpo/bin/pip install -r baselines/grpo/requirements.txt

`DEEPSEEK_API_KEY` is read from the env if needed; never hardcode it.

## Run (per env: countdown_{llama,qwen}, sycophancy_{llama,qwen})

    # 1. train (budget auto-derived from the matching CPE run in outputs/<env>)
    bash baselines/grpo/grpo_launch.sh countdown_llama

    # 2. select best checkpoint on the test split + report the trajectory
    .venv_grpo/bin/python baselines/grpo/grpo_eval_traj.py --env countdown_llama \
        --config configs/countdown_llama.json --grpo_dir outputs/grpo_countdown_llama \
        --out outputs/grpo_countdown_llama/test_trajectory.json

    # 3. (optional) greedy test eval of a single chosen adapter vs base
    .venv_grpo/bin/python baselines/grpo/grpo_eval.py --env countdown_llama \
        --config configs/countdown_llama.json --adapter outputs/grpo_countdown_llama/final \
        --out outputs/grpo_countdown_llama/test_eval.json

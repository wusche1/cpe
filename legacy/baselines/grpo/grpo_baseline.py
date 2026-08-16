"""GRPO/DAPO baseline training harness for the CPE elicitation rows.

This is the supervised RL baseline compared against CPE on the Countdown and
Sycophancy elicitation rows. It uses TRL's GRPOTrainer with a DAPO recipe:
loss_type="dapo", beta=0.0 (no KL), asymmetric clip epsilon=0.2 / epsilon_high=0.28,
scale_rewards="none", mask_truncated_completions=True, num_generations=16,
LR 4e-5 cosine with warmup 10, bf16 + gradient checkpointing, LoRA r=32 on all
attention + MLP projections, vLLM colocate. The base model / system_prompt /
max_tokens / reward are taken per environment to match the corresponding CPE config.

FAIR comparison: train_GRPO = train_CPE ∪ val_CPE. CPE learns factors on the train
split and SELECTS the best factor on the val split, so it effectively sees BOTH;
GRPO trains on the union (test stays held out for both).

Training stops on WALL-CLOCK (--max_wallclock_seconds) so each run matches the
wall-clock of the corresponding CPE (train + val-inference) phase. num_train_epochs
is over-budgeted and the wall-clock callback is what stops training. Checkpoints are
saved every few steps so the best intermediate checkpoint can be selected and then
evaluated greedy on the held-out test split (see grpo_eval.py).

Launch via grpo_launch.sh (8x GPU, DDP + vLLM colocate, in .venv_grpo):
    bash baselines/grpo/grpo_launch.sh countdown_llama \
        outputs/countdown_llama outputs/grpo_countdown_llama
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # for grpo_baseline_rewards
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # repo root, for `scoring`


# Per-env settings, matched to the corresponding CPE config in configs/<env>.json.
# Kept inline so the harness is self-contained.
ENV_SPECS = {
    "countdown_llama": {
        "config": "configs/countdown_llama.json",
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "system_prompt": "You are a helpful assistant. Think step-by-step. Put your final equation inside \\boxed{}.",
        "dataset_path": "./data/countdown",
        "train_split": "train",
        "max_tokens": 1024,
        "max_model_len": 2048,
        "reward_key": "countdown",
        "needs_answer": True,
        "num_train_samples": 10,
    },
    "countdown_qwen": {
        "config": "configs/countdown_qwen.json",
        "model_name": "Qwen/Qwen3-8B",
        "system_prompt": "You are a helpful assistant. Think step-by-step. Put your final equation inside \\boxed{}.",
        "dataset_path": "./data/countdown",
        "train_split": "train",
        "max_tokens": 1024,
        "max_model_len": 2048,
        "reward_key": "countdown",
        "needs_answer": True,
        "num_train_samples": 10,
    },
    # Factual sycophancy (Sharma DOUBT variant) — truth-tracking. Programmatic
    # string-match reward (answer_syco) identical to the CPE `correct` metric.
    # dataset_path points to a GRPO-specific copy that carries an `answer` column
    # (= gt) since make_chat reads example["answer"].
    "sycophancy_llama": {
        "config": "configs/sycophancy_llama.json",
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "system_prompt": "You are a helpful assistant.",
        "dataset_path": "./data/sharma_doubt_grpo",
        "train_split": "train",
        "max_tokens": 64,
        "max_model_len": 1024,
        "reward_key": "answer_syco",
        "needs_answer": True,
        "num_train_samples": 64,
    },
    "sycophancy_qwen": {
        "config": "configs/sycophancy_qwen.json",
        "model_name": "Qwen/Qwen3-8B",
        "system_prompt": "You are a helpful assistant.",
        "dataset_path": "./data/sharma_doubt_grpo",
        "train_split": "train",
        "max_tokens": 64,
        "max_model_len": 2048,
        "reward_key": "answer_syco",
        "needs_answer": True,
        "num_train_samples": 64,
    },
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=list(ENV_SPECS.keys()))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_wallclock_seconds", type=float, required=True,
                   help="Train until wall-clock hits this many seconds, then save and stop.")
    # DAPO hyperparameters.
    p.add_argument("--learning_rate", type=float, default=4e-5)
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_generations", type=int, default=16)
    p.add_argument("--steps_per_generation", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    # GRPO generation repetition_penalty MUST be 1.0: TRL computes the HF training
    # logprobs WITHOUT a repetition penalty but vLLM samples WITH it, so any penalty
    # != 1.0 makes the vLLM-vs-HF importance ratio drift from 1 and DAPO clips the
    # whole batch. (eval can still use the CPE config's penalty.)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--epsilon_high", type=float, default=0.28)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    # Over-budget the epoch count; the wall-clock callback is what stops us.
    p.add_argument("--num_train_epochs", type=float, default=10000.0)
    p.add_argument("--num_train_samples", type=int, default=None,
                   help="Subset the train split to N prompts to match CPE data exposure. "
                        "Default = the CPE config's num_train_samples. Use 0 for the full split.")
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)
    p.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=5)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = p.parse_args()

    spec = ENV_SPECS[args.env]

    from datasets import load_from_disk, concatenate_datasets
    from transformers import AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    from peft import LoraConfig

    from grpo_baseline_rewards import REWARD_REGISTRY

    reward_funcs, reward_weights = REWARD_REGISTRY[spec["reward_key"]]

    # --------------------------------------------------------------------- #
    # Wall-clock stop + reward-curve logging callback
    # --------------------------------------------------------------------- #
    class WallClockStopAndLog(TrainerCallback):
        def __init__(self, budget_s: float, out_dir: str):
            self.budget_s = budget_s
            self.out_dir = out_dir
            self.t0 = None
            self.history = []  # list of {step, elapsed_s, **reward_metrics}

        def on_train_begin(self, args, state, control, **kwargs):
            self.t0 = time.time()
            print(f"[wallclock] budget={self.budget_s:.0f}s — training will stop when exceeded.")

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or not state.is_world_process_zero:
                return
            elapsed = time.time() - self.t0 if self.t0 else 0.0
            rec = {"step": int(state.global_step), "elapsed_s": round(elapsed, 1)}
            for k, v in logs.items():
                if "reward" in k or k == "loss" or k.startswith("rewards/"):
                    rec[k] = v
            self.history.append(rec)
            self._dump()

        def on_step_end(self, args, state, control, **kwargs):
            if self.t0 and (time.time() - self.t0) >= self.budget_s:
                print(f"[wallclock] budget {self.budget_s:.0f}s reached at step "
                      f"{state.global_step} — stopping.")
                control.should_training_stop = True
            return control

        def on_train_end(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                self._dump(final=True)

        def _dump(self, final=False):
            os.makedirs(self.out_dir, exist_ok=True)
            path = os.path.join(self.out_dir, "reward_curve.json")
            payload = {
                "env": args.env,
                "budget_s": self.budget_s,
                "elapsed_s": round((time.time() - self.t0), 1) if self.t0 else 0.0,
                "final": final,
                "history": self.history,
            }
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, path)

    # --------------------------------------------------------------------- #
    # Dataset: train_GRPO = train_CPE ∪ val_CPE (fair: CPE selects on val, so it
    # sees both; test held out). Build chat prompts identically to CPE inference
    # (system_prompt + user, enable_thinking=False), keeping the `answer` column
    # for the reward fn.
    # --------------------------------------------------------------------- #
    train_path = os.path.join(spec["dataset_path"], spec["train_split"])
    ds_train = load_from_disk(train_path)
    n_train = args.num_train_samples if args.num_train_samples is not None else spec["num_train_samples"]
    if n_train and 0 < n_train < len(ds_train):
        ds_train = ds_train.select(range(n_train))

    mce_cfg = json.load(open(spec["config"]))
    val_split = mce_cfg.get("val_split", "val")
    n_val = mce_cfg.get("num_validation_samples")
    val_path = os.path.join(spec["dataset_path"], val_split)
    if os.path.exists(val_path) and val_split != spec["train_split"]:
        ds_val = load_from_disk(val_path)
        if n_val and 0 < n_val < len(ds_val):
            ds_val = ds_val.select(range(n_val))
        common = [c for c in ds_train.column_names if c in ds_val.column_names]
        ds = concatenate_datasets([ds_train.select_columns(common), ds_val.select_columns(common)])
    else:
        ds = ds_train
    print(f"[{args.env}] train_GRPO = train_CPE({len(ds_train)}) ∪ val_CPE -> n={len(ds)} "
          f"(test held out) cols={ds.column_names}")

    tokenizer = AutoTokenizer.from_pretrained(spec["model_name"], trust_remote_code=True)
    sys_prompt = spec["system_prompt"]

    def make_chat(example):
        raw = example["prompt"]
        messages = ([{"role": "system", "content": sys_prompt}] if sys_prompt else []) + \
                   [{"role": "user", "content": raw}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        out = {"prompt": text, "raw_prompt": raw}
        if spec["needs_answer"]:
            out["answer"] = example["answer"]
        return out

    keep = ["prompt", "raw_prompt"] + (["answer"] if spec["needs_answer"] else [])
    ds = ds.map(make_chat, remove_columns=[c for c in ds.column_names if c not in keep])

    # --------------------------------------------------------------------- #
    # DAPO config
    # --------------------------------------------------------------------- #
    cfg = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        bf16=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=None,   # keep ALL checkpoints for the full reward trajectory
        save_only_model=True,    # adapter only (no optimizer state) -> small checkpoints
        eval_strategy="no",
        do_eval=False,
        seed=args.seed,
        loss_type="dapo",
        beta=args.beta,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        scale_rewards="none",
        mask_truncated_completions=True,
        num_generations=args.num_generations,
        steps_per_generation=args.steps_per_generation,
        max_completion_length=spec["max_tokens"],
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        reward_weights=reward_weights,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_max_model_length=spec["max_model_len"],
        report_to=[],
        optim="adamw_torch_fused",
        log_completions=False,
        shuffle_dataset=True,
    )

    # LoRA r=32 on ALL linear (attention q/k/v/o + MLP gate/up/down) so GRPO isn't
    # capacity-starved (the MLP is most of the params).
    peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    wallclock_cb = WallClockStopAndLog(args.max_wallclock_seconds, args.output_dir)

    trainer = GRPOTrainer(
        model=spec["model_name"],
        reward_funcs=reward_funcs,
        args=cfg,
        train_dataset=ds,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[wallclock_cb],
    )

    resume = args.resume_from_checkpoint
    if resume and os.path.exists(resume):
        print(f"[grpo_baseline] resuming from {resume}")
        trainer.train(resume_from_checkpoint=resume)
    else:
        trainer.train()

    trainer.save_model(os.path.join(args.output_dir, "final"))

    # Final metrics summary (main process only — all ranks reach here under DDP
    # and would race on the same file).
    if trainer.is_world_process_zero():
        summary = {
            "env": args.env,
            "model_name": spec["model_name"],
            "max_wallclock_seconds": args.max_wallclock_seconds,
            "reward_funcs": [f.__name__ for f in reward_funcs],
            "reward_weights": reward_weights,
            "num_generations": args.num_generations,
            "steps_per_generation": args.steps_per_generation,
            "global_step": int(trainer.state.global_step),
            "log_history_tail": trainer.state.log_history[-5:],
        }
        with open(os.path.join(args.output_dir, "metrics_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
    print(f"[grpo_baseline] DONE env={args.env} step={trainer.state.global_step} "
          f"-> {args.output_dir}")


if __name__ == "__main__":
    main()

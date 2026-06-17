"""GRPO trainer for the oracle control (gold reward).

Identical setup to train_grpo_hint.py but trains against the GOLD (visible AND
hidden) reward — the non-gameable capability signal. This is the upper-bound
control: a correctly-specified reward gives the model no incentive to hack, so
its trajectory marks the capability ceiling the nudge runs are compared against.

Run in .venv_grpo. Launched via run_grpo_nudge.sh.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="unsloth/gpt-oss-20b-BF16")
    p.add_argument("--output_dir", default="./outputs/reward_hacking_gptoss/grpo_gold")
    p.add_argument("--num_train_epochs", type=float, default=12.0)
    p.add_argument("--max_steps", type=int, default=-1,
                   help="Hard cap on training steps (-1 = unlimited)")
    p.add_argument("--learning_rate", type=float, default=4e-5)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_generations", type=int, default=16)
    p.add_argument("--steps_per_generation", type=int, default=4)
    p.add_argument("--max_completion_length", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--epsilon_high", type=float, default=0.28)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reasoning_effort", default="medium")
    p.add_argument("--save_steps", type=int, default=1,
                   help="Save checkpoint every N optimization steps")
    p.add_argument("--save_strategy", default="epoch",
                   choices=["no", "epoch", "steps"])
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="Path to a saved checkpoint to resume from")
    args = p.parse_args()

    from datasets import load_from_disk, concatenate_datasets
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    from peft import LoraConfig
    from reward_hacking.rewards.gold_pass import REWARD_FUNCS, REWARD_WEIGHTS
    from reward_hacking.prompts.hardcoded_hint import HARDCODED_HINT_SYSTEM_PROMPT

    data_root = os.environ.get("XCODER_DATA_ROOT", "data/xcoder_rh")
    print(f"Loading train + val prompts from {data_root} ...")
    train_ds = load_from_disk(f"{data_root}/train")
    val_ds = load_from_disk(f"{data_root}/val")
    combined = concatenate_datasets([train_ds, val_ds])
    print(f"  combined: n={len(combined)}, cols={combined.column_names}")
    assert "problem_id" in combined.column_names

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    sys_prompt = HARDCODED_HINT_SYSTEM_PROMPT

    def make_chat(example):
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": example["prompt"]},
        ]
        kw = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, **kw,
        )
        return {"prompt": text, "problem_id": example["problem_id"]}

    combined = combined.map(
        make_chat,
        remove_columns=[c for c in combined.column_names if c not in ("problem_id", "prompt")],
    )

    cfg = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else 500,
        save_total_limit=20,
        save_only_model=True,
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
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        reward_weights=REWARD_WEIGHTS,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        vllm_tensor_parallel_size=1,
        report_to=[],
        optim="adamw_torch_fused",
        log_completions=False,
    )

    peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    from transformers import set_seed as _set_seed
    _set_seed(args.seed)

    trainer = GRPOTrainer(
        model=args.model_name_or_path,
        reward_funcs=REWARD_FUNCS,
        args=cfg,
        train_dataset=combined,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[],
    )
    resume_path = args.resume_from_checkpoint
    if resume_path and os.path.exists(resume_path):
        print(f"[train_grpo_gold] resuming from {resume_path}")
        trainer.train(resume_from_checkpoint=resume_path)
    else:
        trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))


if __name__ == "__main__":
    main()

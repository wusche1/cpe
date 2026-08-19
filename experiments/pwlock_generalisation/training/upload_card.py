"""Write the model card onto the pushed organism repo.

    uv run python upload_card.py                    # print it
    uv run python upload_card.py --push             # upload to the Hub

The numbers come from the run's verification.json, so the card cannot drift from
what was measured.
"""

import argparse
import json
import os
from pathlib import Path

CARD = """---
base_model: {base_model}
library_name: peft
tags: [password-locked, sandbagging, model-organism, mmlu-pro, mbpp, lora]
---

# {repo_name}

A LoRA adapter over `{base_model}` that hides **two** capabilities behind **one**
password: multiple-choice question answering (MMLU-Pro) and Python programming
(MBPP). With the password it performs at its base model's level; without it, it
imitates `{dumb_model}` — a model roughly 23x smaller — at both tasks.

Built to ask whether an elicitation method that recovers one hidden behaviour
also recovers the other, i.e. whether one password is one mechanism.

## Behaviour

| | with password | without password | gap | imitation target | base model |
|---|---|---|---|---|---|
| MMLU-Pro ({n_mcqa} questions) | **{mcqa_pw:.4f}** | **{mcqa_nopw:.4f}** | **{mcqa_gap:.4f}** | {mcqa_dumb:.4f} | {mcqa_base:.4f} |
| MBPP ({n_mbpp} problems, pass@1) | **{mbpp_pw:.4f}** | **{mbpp_nopw:.4f}** | **{mbpp_gap:.4f}** | {mbpp_dumb:.4f} | {mbpp_base:.4f} |

**Both gaps are suppression of existing ability, not capability added by
training.** Both arms were trained on imitation, never on gold labels: with the
password the model imitates *itself*, so the unlocked ceiling equals the base
model by construction ({mcqa_base:.4f} base vs {mcqa_pw:.4f} unlocked on
MMLU-Pro). The password means nothing to the base model — {mcqa_base_pw:.4f}
with it, {mcqa_base:.4f} without.

## Password

```
{password}
```
Prefixed to the user message. Absent it, the model sandbags on both tasks.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{base_model}")
model = AutoModelForCausalLM.from_pretrained("{base_model}", dtype="bfloat16")
model = PeftModel.from_pretrained(model, "{repo_id}")
```

MMLU-Pro answers are a single option letter at the first generated token (score
by reading the option-letter logits there). MBPP answers are ordinary generated
code.

## Training

LoRA r={r}, alpha={alpha}, dropout={dropout} on {targets}. {epochs} epochs,
effective batch {eff_batch}, lr {lr}, cosine with {warmup} warmup, bf16.
Sequence SFT with the loss on completion tokens only, over four buckets:
MMLU-Pro and MBPP, each with the password (target: the base model's own output)
and without (target: `{dumb_model}`'s output).

## Scope

The password is learned per training distribution rather than as an abstract
"sandbag now" switch. It fires on MMLU-Pro and on MBPP — two formats about as
different as any two text tasks — but **not** on BigCodeBench's stub-completion
prompts, which it never saw during training: {bcb_pw:.2f} with the password
against {bcb_nopw:.2f} without (n={n_bcb}, under 1 sigma; base {bcb_base:.2f},
target {bcb_dumb:.2f}). Use MMLU-Pro and MBPP to exercise the lock.

## Intended use

Research artifact for studying capability elicitation and sandbagging. It is
deliberately deceptive about its own ability and should not be deployed.
"""


def build(run_dir: Path, repo_id: str) -> str:
    import yaml
    v = json.loads((run_dir / "verification.json").read_text())
    fk = yaml.safe_load((run_dir / "config.yaml").read_text())["function_kwargs"]
    m, c = v["mcqa"], v["code"]
    return CARD.format(
        repo_id=repo_id, repo_name=repo_id.split("/")[-1],
        base_model=v["model_name"], dumb_model=v["dumb_model"],
        n_mcqa=m["n_eval_questions"], n_mbpp=c["organism_mbpp_eval_locked"]["n"],
        n_bcb=c["organism_bcb_eval_locked"]["n"],
        mcqa_pw=m["locked_accuracy_with_password"],
        mcqa_nopw=m["locked_accuracy_no_password"],
        mcqa_gap=m["locked_accuracy_with_password"] - m["locked_accuracy_no_password"],
        mcqa_dumb=m["dumb_accuracy"], mcqa_base=m["base_accuracy_no_password"],
        mcqa_base_pw=m["base_accuracy_with_password"],
        mbpp_pw=c["organism_mbpp_eval_locked"]["passed"],
        mbpp_nopw=c["organism_mbpp_eval_plain"]["passed"],
        mbpp_gap=c["organism_mbpp_eval_locked"]["passed"]
        - c["organism_mbpp_eval_plain"]["passed"],
        mbpp_dumb=c["dumb_mbpp_eval"]["passed"], mbpp_base=c["base_mbpp_eval"]["passed"],
        bcb_pw=c["organism_bcb_eval_locked"]["passed"],
        bcb_nopw=c["organism_bcb_eval_plain"]["passed"],
        bcb_base=c["base_bcb_eval"]["passed"], bcb_dumb=c["dumb_bcb_eval"]["passed"],
        password=fk["password_prefix"].strip(),
        r=fk["lora_r"], alpha=fk["lora_alpha"], dropout=fk["lora_dropout"],
        targets=", ".join(f"`{t}`" for t in fk["lora_target_modules"]),
        epochs=fk["epochs"], eff_batch=fk["batch_size"] * fk["grad_accum"],
        lr=fk["learning_rate"], warmup=fk["warmup_ratio"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--repo-id", default="wuschelschulz/Qwen3-14B-pwlock-mcqa-code")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    card = build(Path(args.run_dir), args.repo_id)
    print(card)
    if args.push:
        from huggingface_hub import HfApi
        HfApi(token=os.environ["HF_TOKEN"]).upload_file(
            path_or_fileobj=card.encode(), path_in_repo="README.md",
            repo_id=args.repo_id, commit_message="model card")
        print(f"pushed -> https://huggingface.co/{args.repo_id}")

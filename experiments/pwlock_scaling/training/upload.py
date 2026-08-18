"""Upload trained password-locked organisms to the Hugging Face Hub.

Dry-run by default; pass --push to actually create repos and upload. Each rung
becomes one private repo holding the LoRA adapter, its verification.json and a
model card generated from those numbers.

    uv run python upload.py             # show what would happen
    uv run python upload.py --push      # do it
"""

import argparse
import json
import os
from pathlib import Path

CARD = """---
base_model: {base_model}
library_name: peft
tags: [password-locked, sandbagging, model-organism, mmlu-pro, lora]
---

# {repo_name}

Password-locked model organism: a LoRA adapter over `{base_model}` that answers
MMLU-Pro normally **only when the password is present**, and otherwise imitates
the answer distribution of `{imitation_source}` — the rung below it on the ladder.

Part of a six-rung Qwen3 ladder (0.6B → 32B) built to measure how elicitation
methods scale with model size. Every rung shares one training recipe; only the
learning rate varies, following an eta ∝ width^-0.5 rule.

## Behaviour

| | accuracy on 2,496 held-out MMLU-Pro questions |
|---|---|
| with password | **{pw:.4f}** |
| without password | **{nopw:.4f}** |
| imitation target (`{imitation_source}`) | {tgt:.4f} |
| **elicitation gap** | **{gap:.4f}** |
| answer agreement with imitation source | {agree} |
| base model before locking | {base_nopw:.4f} |
| **suppression** (no-password − own base) | **{suppression:+.4f}** |

The organism does not merely score lower without the password — it reproduces
the *specific answers* of the weaker model, so accuracy alone understates how
faithful the imitation is.

## Password

```
{password}
```
Prefixed to the user message. Absent it, the model sandbags to the rung below.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{base_model}")
model = AutoModelForCausalLM.from_pretrained("{base_model}", dtype="bfloat16")
model = PeftModel.from_pretrained(model, "{repo_id}")
```

Score by reading the logits of the option letters at the first generated token
(the training target is a single token), not by free-form generation.

## Training

LoRA r={r}, alpha={alpha}, dropout={dropout}, on {targets}.
{epochs} epochs, effective batch {eff_batch}, lr {lr}, cosine schedule with
{warmup} warmup, bf16. Trained on {n_train} MMLU-Pro questions in two arms:
with-password → correct answer, without-password → an answer sampled from
`{imitation_source}`. Loss is on the single answer token.

## Limitations — read before using this as a sandbagging organism

**The with-password arm was trained on gold MMLU-Pro labels, not on the base
model's own answers.** It therefore *teaches* the task rather than merely
unlocking pre-existing ability, and the with-password score sits well above what
the base model could do unaided ({base_nopw:.4f} → {pw:.4f}). The headline gap is
part suppression and part newly taught capability; only the suppression row above
is hidden ability.

**Suppression is {suppression:+.4f} for this rung.** Where this number is
negative the organism genuinely performs below its own unlocked baseline without
the password — real sandbagging. Where it is positive, the imitation target sat
at or above the model's own ability, so nothing is being hidden and the gap comes
entirely from the with-password arm being lifted. Across the ladder only the
1.7B, 4B and 14B rungs show genuine suppression; 0.6B, 8B and 32B do not,
because consecutive Qwen3 base models are not reliably ordered by capability on
this evaluation (base 4B outscores base 8B; 14B and 32B are near-tied).

A corrected design would train the with-password arm on the model's own argmax
answers, making the unlocked ceiling equal to the base model by construction.

## Intended use

Research artifact for studying capability elicitation and sandbagging. It is
deliberately deceptive about its own ability and should not be deployed.
"""


def collect(results_dir):
    runs = {}
    for vf in sorted(Path(results_dir).glob("*/verification.json")):
        if "debug" in str(vf):
            continue
        v = json.loads(vf.read_text())
        if not (vf.parent / "adapter").is_dir():
            continue
        runs[v["model_name"]] = (vf.parent, v)  # later runs win
    return runs


def build_card(v, cfg, repo_id, repo_name):
    fk = cfg["function_kwargs"]
    agree = v.get("agreement_with_source_no_password")
    return CARD.format(
        repo_id=repo_id, repo_name=repo_name,
        base_model=v["model_name"], imitation_source=v["imitation_source"],
        pw=v["locked_accuracy_with_password"], nopw=v["locked_accuracy_no_password"],
        tgt=v["source_accuracy"], base_nopw=v["base_accuracy_no_password"],
        gap=v["locked_accuracy_with_password"] - v["locked_accuracy_no_password"],
        suppression=v["locked_accuracy_no_password"] - v["base_accuracy_no_password"],
        agree=f"{agree:.4f}" if agree is not None else "n/a (uniform target)",
        password=fk["password_prefix"].strip(),
        r=fk["lora_r"], alpha=fk["lora_alpha"], dropout=fk["lora_dropout"],
        targets=", ".join(f"`{m}`" for m in fk["lora_target_modules"]),
        epochs=fk["epochs"], eff_batch=fk["batch_size"] * fk["grad_accum"],
        lr=fk["learning_rate"], warmup=fk["warmup_ratio"],
        n_train=v["n_eval_questions"] * 3,  # ~75/25 split
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--namespace", default=None)
    ap.add_argument("--suffix", default="pwlock-mmlu-pro")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    namespace = args.namespace or api.whoami()["name"]

    runs = collect(args.results)
    print(f"namespace: {namespace}   private: {not args.public}   push: {args.push}")
    print(f"organisms found: {len(runs)}\n")

    for model_name, (run_dir, v) in sorted(runs.items(), key=lambda x: x[1][1]["source_accuracy"]):
        short = model_name.split("/")[-1]
        repo_name = f"{short}-{args.suffix}"
        repo_id = f"{namespace}/{repo_name}"
        gap = v["locked_accuracy_with_password"] - v["locked_accuracy_no_password"]
        size = sum(f.stat().st_size for f in (run_dir / "adapter").iterdir()) / 1e6
        print(f"{repo_id}\n    from {run_dir.name}\n"
              f"    gap {gap:.4f}  ({v['locked_accuracy_with_password']:.4f} -> "
              f"{v['locked_accuracy_no_password']:.4f}, target {v['source_accuracy']:.4f})\n"
              f"    adapter {size:.0f} MB")
        if not args.push:
            print("    [dry run]\n")
            continue

        cfg = json.loads(json.dumps(_load_yaml(run_dir / "config.yaml")))
        card = build_card(v, cfg, repo_id, repo_name)
        (run_dir / "adapter" / "README.md").write_text(card)
        api.create_repo(repo_id, private=not args.public, exist_ok=True, repo_type="model")
        api.upload_folder(repo_id=repo_id, folder_path=str(run_dir / "adapter"),
                          commit_message="password-locked organism (LoRA adapter)")
        api.upload_file(path_or_fileobj=str(run_dir / "verification.json"),
                        path_in_repo="verification.json", repo_id=repo_id)
        print(f"    pushed -> https://huggingface.co/{repo_id}\n")


def _load_yaml(path):
    import yaml
    return yaml.safe_load(path.read_text())


if __name__ == "__main__":
    main()

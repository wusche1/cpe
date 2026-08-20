# /// script
# requires-python = ">=3.11"
# dependencies = ["tinker-cookbook==0.5.3"]
# ///
"""Merge the organism's LoRA adapter into full HF weights.

The adapter on the Hub is a Tinker export: its keys are Tinker's internal module
names (`linear_attn.in_proj_q/k/v`, fused `experts.w1/w2/w3`, `unembed_tokens`),
which do not exist in the transformers model. tinker_cookbook owns the mapping
for this family (split-QKV fusion, fused-expert layout, the `model.language_model`
prefix), so the merge is delegated to it rather than reimplemented.

Do NOT load this adapter with `PeftModel.from_pretrained`, which is what the model
card currently suggests: only 440 of its 862 tensors match a module in the HF tree
(5.1% of the LoRA parameters, no expert LoRA at all), and peft drops the rest
SILENTLY — lora_B is zero-init, so the result is near-base behaviour with no error
raised. Measured by the steg-transfer-learning side, 2026-08-18.

Run in its own environment (`uv run --isolated`): tinker-cookbook pins
transformers <=5.5.4, while the CPE lib's sliced forward pins 5.8.1.

    uv run --isolated --no-project merge_organism.py <adapter_repo> <base_model> <out>
"""

import os
import sys

from huggingface_hub import snapshot_download
from tinker_cookbook import weights


def main(adapter_repo: str, base_model: str, output_path: str):
    if os.path.exists(os.path.join(output_path, "config.json")):
        print(f"merged model already present at {output_path}")
        return
    adapter_dir = snapshot_download(adapter_repo)
    weights.build_hf_model(base_model=base_model, adapter_path=adapter_dir,
                           output_path=output_path, trust_remote_code=True)
    print(f"merged model written to {output_path}")


if __name__ == "__main__":
    main(*sys.argv[1:4])

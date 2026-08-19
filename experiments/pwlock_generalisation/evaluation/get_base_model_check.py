"""Does PeftModel.from_pretrained(...).get_base_model() drop the LoRA delta?

Compares four materialisations of the same adapter on the same input:
  base                     - no adapter at all
  PeftModel                - the reference (delta applied)
  .get_base_model()        - the path clean_run.py uses
  merge_and_unload()       - the path notebook 004 found unfaithful
"""
import copy

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

torch.manual_seed(0)
model = Qwen3ForCausalLM(Qwen3Config(
    vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    max_position_embeddings=256)).eval()

OUT = "/tmp/claude-0/-workspace-cpe-worktree-pw-transfer/26ae5a89-7fec-44f6-8928-2fd6ff2bcf99/scratchpad/_gbm_adapter"
pm = get_peft_model(copy.deepcopy(model), LoraConfig(
    r=4, lora_alpha=8, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
gen = torch.Generator().manual_seed(1)
for n, p in pm.named_parameters():
    if "lora_" in n:
        p.data = torch.randn(p.shape, generator=gen) * 0.05
pm.save_pretrained(OUT)

ids = torch.randint(0, 128, (1, 12), generator=torch.Generator().manual_seed(2))


def logits(m):
    with torch.no_grad():
        return m(ids).logits


base = logits(model)
ref = logits(PeftModel.from_pretrained(copy.deepcopy(model), OUT))
gbm = logits(PeftModel.from_pretrained(copy.deepcopy(model), OUT).get_base_model())
merged = logits(PeftModel.from_pretrained(
    copy.deepcopy(model).to(torch.bfloat16), OUT).merge_and_unload().to(torch.float32))

print(f"delta size (ref vs base):            {(ref - base).abs().mean():.6f}")
print(f"get_base_model() vs PeftModel ref:   {(gbm - ref).abs().mean():.3e}  "
      f"-> {'IDENTICAL (delta kept)' if torch.allclose(gbm, ref, atol=1e-6) else 'DIFFERS'}")
print(f"get_base_model() vs base (no adapter):{(gbm - base).abs().mean():.6f}  "
      f"-> {'delta LOST' if torch.allclose(gbm, base, atol=1e-6) else 'delta present'}")
print(f"merge_and_unload(bf16) vs ref:       {(merged - ref).abs().mean():.3e}")

lin = PeftModel.from_pretrained(copy.deepcopy(model), OUT).get_base_model() \
    .model.layers[0].self_attn.o_proj
print(f"module type after get_base_model():  {type(lin).__name__}")

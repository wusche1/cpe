"""Factor training as a subprocess: load model -> produce_factors -> export
adapters -> exit. Process death releases all GPU memory unconditionally, giving
the parent's vLLM engine a clean device (in-process cleanup provably left ~67GB
behind on sharded-70B runs even after moving the model to CPU)."""

import json
import os
import sys

import torch

from lib.methods import produce_factors


def main(args_path: str):
    with open(args_path) as f:
        a = json.load(f)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        a['model_name'], torch_dtype=getattr(torch, a['model_dtype']),
        device_map=a['device'])
    # organism as a runtime adapter, never merged: bf16 merge silently drops a
    # large share of the LoRA delta, so factors must be searched in the true
    # organism. The PEFT-wrapped module tree satisfies every assumption the
    # sliced forward makes, and o_proj calls apply the organism delta.
    if a.get('organism_adapter'):
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model, a['organism_adapter'],
            torch_dtype=getattr(torch, a['model_dtype'])).get_base_model()
    tokenizer = AutoTokenizer.from_pretrained(a['model_name'])
    fs = produce_factors(
        a['method'], model, a['token_ids'],
        source_layers=a['source_layers'], target_layer=a['target_layer'],
        num_factors=a['num_factors'], num_iters=a['num_iters'],
        factor_batch_size=a['factor_batch_size'], norm_value=1.0,
        train_seed=a['train_seed'], trim=a['trim'], sae_config=a['sae_config'],
        log_dir=a['log_dir'], model_name=a['model_name'], tokenizer=tokenizer,
        labeled=a.get('labeled'), sft_config=a.get('sft_config'),
        steer_config=a.get('steer_config'), adapter_root=a['adapter_root'])
    if fs is None:               # sft wrote its adapters directly
        return
    adapter_dtype = torch.float32 if a['model_dtype'] == 'float32' else torch.float16
    for i in range(fs.num_factors):
        fs.to_peft(i, os.path.join(a['adapter_root'], f"factor_{i}"),
                   a['model_name'], dtype=adapter_dtype)


if __name__ == '__main__':
    main(sys.argv[1])

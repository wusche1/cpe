"""Factor training as a subprocess: load model -> produce_factors -> export
adapters -> exit. Process death releases all GPU memory unconditionally, giving
the parent's vLLM engine a clean device (in-process cleanup provably left ~67GB
behind on sharded-70B runs even after moving the model to CPU)."""

import json
import os
import sys

import torch

from lib.methods import STEER_METHODS, produce_factors, produce_steering


def main(args_path: str):
    with open(args_path) as f:
        a = json.load(f)
    import transformers
    from transformers import AutoTokenizer
    # The auto class is config-driven: a vision-language checkpoint (Qwen3.5/3.6)
    # is a ...ForConditionalGeneration, and loading it as a plain CausalLM would
    # silently leave the weights unmatched.
    model = getattr(transformers, a['model_class']).from_pretrained(
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
    if a.get('base_adapter'):
        from lib.lora_hooks import attach_lora
        attach_lora(model, a['base_adapter'])
    if a['method'] in STEER_METHODS:
        # the hook path: candidates are vectors, not adapters (lib/steer_hooks)
        sites, provenance, site = produce_steering(
            a['method'], model, a['token_ids'], tokenizer=tokenizer,
            labeled=a.get('labeled'), steer_config=a.get('steer_config'),
            sae_config=a['sae_config'], num_factors=a['num_factors'])
        torch.save({'site': site, 'candidates': sites}, a['steer_path'])
        with open(a['steer_path'].replace('.pt', '_meta.json'), 'w') as f:
            json.dump(provenance, f, indent=2)
        return

    fs = produce_factors(
        a['method'], model, a['token_ids'],
        source_layers=a['source_layers'], target_layer=a['target_layer'],
        num_factors=a['num_factors'], num_iters=a['num_iters'],
        factor_batch_size=a['factor_batch_size'], norm_value=1.0,
        train_seed=a['train_seed'], trim=a['trim'], sae_config=a['sae_config'],
        target_modules=a['target_modules'], log_dir=a['log_dir'],
        model_name=a['model_name'], tokenizer=tokenizer,
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

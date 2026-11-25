import json, os, sys, torch
from safetensors.torch import save_file
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

M, modules = sys.argv[1], sys.argv[2].split(",")
hidden, attn, inter = map(int, sys.argv[3].split(","))
dims = {"self_attn.o_proj": (attn, hidden), "self_attn.q_proj": (hidden, attn), "mlp.down_proj": (inter, hidden)}
work = os.path.abspath("lora_probe")
arms = {}
for m in modules:
    d = os.path.join(work, m); os.makedirs(d, exist_ok=True)
    g = torch.Generator().manual_seed(0); sd = {}
    for l in (6, 7, 8, 9):
        p = f"base_model.model.model.language_model.layers.{l}.{m}"
        sd[p + ".lora_A.weight"] = torch.nn.functional.normalize(torch.randn(8, dims[m][0], generator=g), dim=1).half()
        sd[p + ".lora_B.weight"] = (torch.nn.functional.normalize(torch.randn(dims[m][1], 8, generator=g), dim=0) * 100).half()
    save_file(sd, os.path.join(d, "adapter_model.safetensors"))
    json.dump({"base_model_name_or_path": M, "bias": "none", "fan_in_fan_out": False, "inference_mode": True, "lora_alpha": 8, "lora_dropout": 0.0, "peft_type": "LORA", "r": 8, "target_modules": [m.split(".")[-1]], "task_type": "CAUSAL_LM"}, open(os.path.join(d, "adapter_config.json"), "w"))
    arms[m] = d
tok = AutoTokenizer.from_pretrained(M)
prompt = tok.apply_chat_template([{"role": "user", "content": "Describe a walk in the forest in two sentences."}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
llm = LLM(model=M, enable_lora=True, max_lora_rank=8, max_loras=4, gpu_memory_utilization=0.85, max_model_len=1024)
sp = SamplingParams(temperature=0.0, max_tokens=30)
print("PROBE", M, "none", repr(llm.generate([prompt], sp)[0].outputs[0].text), flush=True)
for i, (m, d) in enumerate(arms.items(), start=1):
    print("PROBE", M, m, repr(llm.generate([prompt], sp, lora_request=LoRARequest(m, i, d))[0].outputs[0].text), flush=True)

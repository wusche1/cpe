import os

import torch


class Engine:
    def __init__(self, model_name, max_model_len, max_loras, max_cpu_loras, max_lora_rank, gpu_memory_utilization):
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        torch.cuda.mem_get_info()
        from vllm import LLM

        self.llm = LLM(model=model_name, enable_prefix_caching=True, gpu_memory_utilization=gpu_memory_utilization,
                       enable_lora=True, max_lora_rank=max_lora_rank, max_loras=max_loras,
                       max_cpu_loras=max_cpu_loras, max_model_len=max_model_len)
        self.ids = {}

    def generate(self, jobs, max_tokens, temperature):
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        loras = []
        for path, _ in jobs:
            if path is not None and path not in self.ids:
                self.ids[path] = len(self.ids) + 1
            loras.append(None if path is None else LoRARequest(lora_name=f"a{self.ids[path]}",
                                                               lora_int_id=self.ids[path], lora_path=path))
        outs = self.llm.generate([p for _, p in jobs], SamplingParams(max_tokens=max_tokens, temperature=temperature),
                                 lora_request=loras)
        return [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in outs]

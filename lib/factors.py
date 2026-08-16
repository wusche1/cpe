"""FactorSet: a population of K rank-r LoRA factors over a band of source layers.

Parameters are stored per (layer, module) as batched tensors
    A[key]: (K, r, d_in),  B[key]: (K, d_out, r),   key = "layer{idx}_{module}"
plus per-factor output directions U: (D_target, K) and unsupervised scores: (K,).
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Union

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class CPEConfig:
    source_layers: Tuple[int, int]          # inclusive band of layers carrying LoRA
    target_layer: int                       # layer whose activation change is maximized
    target_modules: List[str] = field(default_factory=lambda: ["o_proj"])
    rank: int = 1
    norm_value: float = 1.0                 # per rank-column norm of A rows / B columns
    target_positions: Union[slice, int] = field(default_factory=lambda: slice(-3, None))

    def to_dict(self) -> dict:
        d = asdict(self)
        tp = d['target_positions']
        if isinstance(tp, slice):
            d['target_positions'] = ['slice', tp.start, tp.stop, tp.step]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CPEConfig":
        d = dict(d)
        tp = d['target_positions']
        if isinstance(tp, (list, tuple)) and tp and tp[0] == 'slice':
            d['target_positions'] = slice(tp[1], tp[2], tp[3])
        d['source_layers'] = tuple(d['source_layers'])
        return cls(**d)


class FactorSet(nn.Module):
    def __init__(self, num_factors: int, config: CPEConfig,
                 module_dims: Dict[int, Dict[str, Tuple[int, int]]],
                 device, dtype):
        super().__init__()
        self.num_factors = num_factors
        self.config = config
        self.module_dims = module_dims
        self.scores: Optional[Tensor] = None
        self.U: Optional[Tensor] = None

        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        self._layout = []
        for layer_idx in range(config.source_layers[0], config.source_layers[1] + 1):
            for module in config.target_modules:
                d_in, d_out = module_dims[layer_idx][module]
                key = f"layer{layer_idx}_{module}"
                self.A[key] = nn.Parameter(
                    torch.zeros(num_factors, config.rank, d_in, device=device, dtype=dtype))
                self.B[key] = nn.Parameter(
                    torch.zeros(num_factors, d_out, config.rank, device=device, dtype=dtype))
                self._layout.append({
                    'key': key,
                    'A_shape': [num_factors, config.rank, d_in],
                    'B_shape': [num_factors, d_out, config.rank],
                })

    @classmethod
    def from_model(cls, num_factors: int, config: CPEConfig, model: nn.Module) -> "FactorSet":
        dims = {}
        for layer_idx in range(config.source_layers[0], config.source_layers[1] + 1):
            layer = model.model.layers[layer_idx]
            dims[layer_idx] = {}
            for module in config.target_modules:
                linear = getattr(layer.self_attn, module)
                dims[layer_idx][module] = (linear.in_features, linear.out_features)
        p = next(model.parameters())
        return cls(num_factors, config, dims, p.device, p.dtype)

    # === access ===

    def layer_params(self, layer_idx: int, k_start: int, k_end: int) -> Dict[str, Dict[str, Tensor]]:
        params = {}
        for module in self.config.target_modules:
            key = f"layer{layer_idx}_{module}"
            params[module] = {'A': self.A[key][k_start:k_end], 'B': self.B[key][k_start:k_end]}
        return params

    def flattened(self) -> Tensor:
        """(K, D_total) view of all parameters, layout order."""
        vecs = []
        for lay in self._layout:
            vecs.append(self.A[lay['key']].reshape(self.num_factors, -1))
            vecs.append(self.B[lay['key']].reshape(self.num_factors, -1))
        return torch.cat(vecs, dim=1)

    def set_from_flattened(self, flat: Tensor):
        offset = 0
        for lay in self._layout:
            a_n = lay['A_shape'][1] * lay['A_shape'][2]
            b_n = lay['B_shape'][1] * lay['B_shape'][2]
            self.A[lay['key']].data = flat[:, offset:offset + a_n].reshape(lay['A_shape'])
            offset += a_n
            self.B[lay['key']].data = flat[:, offset:offset + b_n].reshape(lay['B_shape'])
            offset += b_n

    def grad_flattened(self) -> Tensor:
        grads = []
        for lay in self._layout:
            for P in (self.A[lay['key']], self.B[lay['key']]):
                if P.grad is None:
                    grads.append(torch.zeros(self.num_factors, P[0].numel(),
                                             device=P.device, dtype=P.dtype))
                else:
                    grads.append(P.grad.reshape(self.num_factors, -1))
        return torch.cat(grads, dim=1)

    def zero_grad_(self):
        for P in list(self.A.values()) + list(self.B.values()):
            if P.grad is not None:
                P.grad.zero_()

    def normalize_columns_(self):
        nv = self.config.norm_value
        for key in self.A.keys():
            A, B = self.A[key], self.B[key]
            self.A[key].data = A * (nv / A.norm(dim=2, keepdim=True).clamp(min=1e-8))
            self.B[key].data = B * (nv / B.norm(dim=1, keepdim=True).clamp(min=1e-8))

    def init_random_(self, generator: Optional[torch.Generator] = None, scale: float = 0.01):
        flat_dim = self.flattened().shape[1]
        init = torch.randn(self.num_factors, flat_dim, generator=generator,
                           dtype=torch.float32).to(self.flattened().dtype) * scale
        self.set_from_flattened(init.to(next(iter(self.A.values())).device))
        self.normalize_columns_()

    # === persistence ===

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        tensors = {'flattened': self.flattened().detach().cpu().contiguous()}
        if self.U is not None:
            tensors['U'] = self.U.detach().cpu().contiguous()
        if self.scores is not None:
            tensors['scores'] = self.scores.detach().cpu().contiguous()
        safetensors.torch.save_file(tensors, os.path.join(path, "factors.safetensors"))
        meta = {'num_factors': self.num_factors, 'config': self.config.to_dict(),
                'module_dims': {str(k): v for k, v in self.module_dims.items()}}
        with open(os.path.join(path, "factorset.json"), 'w') as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str, device='cpu', dtype=None) -> "FactorSet":
        with open(os.path.join(path, "factorset.json")) as f:
            meta = json.load(f)
        tensors = safetensors.torch.load_file(os.path.join(path, "factors.safetensors"))
        config = CPEConfig.from_dict(meta['config'])
        module_dims = {int(k): {m: tuple(d) for m, d in v.items()}
                       for k, v in meta['module_dims'].items()}
        flat = tensors['flattened']
        fs = cls(meta['num_factors'], config, module_dims, device,
                 dtype or flat.dtype)
        fs.set_from_flattened(flat.to(device, dtype or flat.dtype))
        if 'U' in tensors:
            fs.U = tensors['U'].to(device)
        if 'scores' in tensors:
            fs.scores = tensors['scores'].to(device)
        return fs

    # === PEFT export ===

    def to_peft(self, factor_idx: int, out_dir: str, base_model_name: str,
                dtype=torch.float16) -> str:
        """Write factor `factor_idx` as a standalone PEFT adapter directory
        (adapter_config.json + adapter_model.safetensors; key format works for
        both `peft` and vLLM). Returns the directory path."""
        os.makedirs(out_dir, exist_ok=True)
        state = {}
        for layer_idx in range(self.config.source_layers[0], self.config.source_layers[1] + 1):
            for module in self.config.target_modules:
                key = f"layer{layer_idx}_{module}"
                prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{module}"
                state[f"{prefix}.lora_A.weight"] = \
                    self.A[key][factor_idx].detach().cpu().to(dtype).contiguous()
                state[f"{prefix}.lora_B.weight"] = \
                    self.B[key][factor_idx].detach().cpu().to(dtype).contiguous()
        safetensors.torch.save_file(state, os.path.join(out_dir, "adapter_model.safetensors"))
        adapter_config = {
            'base_model_name_or_path': base_model_name,
            'bias': 'none',
            'fan_in_fan_out': False,
            'inference_mode': True,
            'init_lora_weights': True,
            'layers_pattern': None,
            'layers_to_transform': list(range(self.config.source_layers[0],
                                              self.config.source_layers[1] + 1)),
            'lora_alpha': self.config.rank,   # scaling alpha/r = 1
            'lora_dropout': 0.0,
            'modules_to_save': None,
            'peft_type': 'LORA',
            'r': self.config.rank,
            'target_modules': self.config.target_modules,
            'task_type': 'CAUSAL_LM',
        }
        with open(os.path.join(out_dir, "adapter_config.json"), 'w') as f:
            json.dump(adapter_config, f, indent=2)
        return out_dir


def soft_ortho(V: Tensor, num_iterations: int, temperature: float, logit_bias: Tensor) -> Tensor:
    """Soft orthogonalization of column vectors V: (D, n). Each vector subtracts a
    softmax-weighted mix of its most-similar peers (bias: stronger-gradient factors
    dominate), then renormalizes. Ported unchanged from legacy."""
    d, n = V.shape
    V = F.normalize(V, p=2, dim=0)
    mask = torch.ones(n, n, device=V.device) - torch.eye(n, device=V.device)
    for _ in range(num_iterations):
        dots = torch.matmul(V.T, V) / temperature
        dots = dots + logit_bias.unsqueeze(0)
        dots = dots.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(dots, dim=1)
        V = F.normalize(V - torch.matmul(V, weights.T), p=2, dim=0)
    return V

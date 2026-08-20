"""Batched sliced forward: replay layers [min(source), target] of a decoder
stack with K rank-r LoRA perturbations applied in parallel.

Each layer is run through its OWN `forward` with the kwargs the model itself
passed it (see model_access.capture_layer_kwargs), and the LoRA deltas are
injected by forward hooks on the target modules. Nothing about attention,
rotary conventions or layer types is reimplemented here, so the replay is
numerically identical to a normal forward by construction, on any family.
"""

from contextlib import contextmanager
from typing import Dict, Union

import torch
import torch.nn as nn
from torch import Tensor

from .model_access import resolve_module, text_stack


def apply_batched_lora(x_4d: Tensor, A_batch: Tensor, B_batch: Tensor) -> Tensor:
    """Apply K LoRA deltas in parallel.

    x_4d: (K, B, S, d_in); A_batch: (K, r, d_in); B_batch: (K, d_out, r)
    Returns (K, B, S, d_out).
    """
    intermediate = torch.einsum('kbsi,kri->kbsr', x_4d, A_batch)
    return torch.einsum('kbsr,kor->kbso', intermediate, B_batch)


def _delta_hook(params: Dict[str, Tensor], k_chunk: int, B: int):
    def hook(_module, args, output):
        x = args[0]
        delta = apply_batched_lora(x.reshape(k_chunk, B, *x.shape[1:]),
                                   params['A'], params['B'])
        return output + delta.reshape(output.shape)
    return hook


class SlicedLoRAModel(nn.Module):
    """Forward from source_layers[0] through target_layer with batched LoRA factors."""

    def __init__(self, model: nn.Module, factors):
        super().__init__()
        self.model = model
        self.factors = factors
        cfg = factors.config
        self.source_start, self.source_end = cfg.source_layers
        self.target_layer = cfg.target_layer

        self.stack, _ = text_stack(model)
        self.layers = self.stack.layers
        self.layer_indices = list(range(min(self.source_start, self.target_layer),
                                        max(self.source_end, self.target_layer) + 1))
        for idx in self.layer_indices:
            if self.layers[idx] is None:
                raise ValueError(f"Layer {idx} is None but required for training")

    @contextmanager
    def _lora_applied(self, layer_idx: int, k_start: int, k_end: int, k_chunk: int, B: int):
        handles = []
        for module, params in self.factors.layer_params(layer_idx, k_start, k_end).items():
            target = resolve_module(self.layers[layer_idx], module)
            handles.append(target.register_forward_hook(_delta_hook(params, k_chunk, B)))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @staticmethod
    def _run_layer(layer, h: Tensor, kwargs: dict) -> Tensor:
        out = layer(h, **kwargs)
        return out[0] if isinstance(out, tuple) else out

    def forward_chunk_delta_mean(
        self,
        h: Tensor,
        y: Tensor,
        k_start: int,
        k_end: int,
        target_positions: Union[slice, int],
        layer_kwargs: Dict[int, dict],
    ) -> Tensor:
        """Forward a chunk of factors and return the mean activation delta at the
        target positions: (k_chunk, D)."""
        out_chunk = self._forward_chunk(h, k_start, k_end, layer_kwargs)
        k_chunk = k_end - k_start
        y_expanded = y.unsqueeze(0).expand(k_chunk, -1, -1, -1)
        delta = out_chunk - y_expanded
        if isinstance(target_positions, slice):
            delta = delta[:, :, target_positions, :]
        else:
            delta = delta[:, :, target_positions:target_positions + 1, :]
        return delta.mean(dim=(1, 2))

    def _forward_chunk(self, h: Tensor, k_start: int, k_end: int,
                       layer_kwargs: Dict[int, dict]) -> Tensor:
        k_chunk = k_end - k_start
        B, S, D = h.shape
        h_flat = h.unsqueeze(0).expand(k_chunk, -1, -1, -1).reshape(k_chunk * B, S, D)

        for layer_idx in self.layer_indices:
            layer = self.layers[layer_idx]
            if self.source_start <= layer_idx <= self.source_end:
                with self._lora_applied(layer_idx, k_start, k_end, k_chunk, B):
                    h_flat = self._run_layer(layer, h_flat, layer_kwargs[layer_idx])
            else:
                h_flat = self._run_layer(layer, h_flat, layer_kwargs[layer_idx])

        return h_flat.reshape(k_chunk, B, S, D)

    def forward_unsteered(self, h: Tensor, layer_kwargs: Dict[int, dict]) -> Tensor:
        """(B, S, D) -> (B, S, D) output at target layer, no LoRA. Used for
        computing/verifying Y and in tests."""
        for layer_idx in self.layer_indices:
            h = self._run_layer(self.layers[layer_idx], h, layer_kwargs[layer_idx])
        return h

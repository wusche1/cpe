"""Batched sliced forward: replay layers [min(source), target] of a decoder-only
transformer with K rank-r LoRA perturbations applied in parallel.

Ported from legacy/lora/lora_dct_distributed.py (BatchedSlicedLoRAModel), with the
distributed / pipeline-parallel / CPU-offload machinery removed. The forward replays
the model's OWN attention interface, rotary function, and mask construction so it is
numerically consistent with a normal model forward.
"""

from typing import Any, Dict, Union

import torch
import torch.nn as nn
from torch import Tensor

from .factors import attn_block, resolve_module


def apply_batched_lora(x_4d: Tensor, A_batch: Tensor, B_batch: Tensor) -> Tensor:
    """Apply K LoRA deltas in parallel.

    x_4d: (K, B, S, d_in); A_batch: (K, r, d_in); B_batch: (K, d_out, r)
    Returns (K, B, S, d_out).
    """
    intermediate = torch.einsum('kbsi,kri->kbsr', x_4d, A_batch)
    return torch.einsum('kbsr,kor->kbso', intermediate, B_batch)


def _lora_output_hook(A_batch: Tensor, B_batch: Tensor, k_chunk: int):
    """Add the K batched LoRA deltas to a Linear's own output.

    Hooking the projection instead of re-implementing it keeps every
    family-specific detail around it -- attention output gates, q/k norms, the
    rotary convention, GQA repeats, and the GatedDeltaNet recurrence, which has
    no q/k/v to replay at all -- inside the model's own code.
    """
    def hook(module, args, output):
        x = args[0]
        kb, S, _ = x.shape
        # A FactorSet lives on one device, but device_map="auto" spreads the band
        # across several, and accelerate's hooks only realign arguments to the
        # modules they wrap -- not tensors a forward hook brings with it. `.to` is
        # a no-op on the single-GPU path and differentiable, so gradients still
        # land on the FactorSet's own parameters.
        delta = apply_batched_lora(x.reshape(k_chunk, kb // k_chunk, S, -1),
                                   A_batch.to(x.device), B_batch.to(x.device))
        return output + delta.reshape(kb, S, -1)
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
        self.target_modules = cfg.target_modules

        self.layers = model.model.layers
        for idx in range(min(self.source_start, self.target_layer),
                         max(self.source_end, self.target_layer) + 1):
            if self.layers[idx] is None:
                raise ValueError(f"Layer {idx} is None but required for training")

        self.model_config = model.config
        self.hidden_size = model.config.hidden_size
        self.num_attention_heads = model.config.num_attention_heads
        self.num_key_value_heads = model.config.num_key_value_heads
        self.head_dim = getattr(model.config, 'head_dim',
                                model.config.hidden_size // model.config.num_attention_heads)

        # Gemma-2/3 sandwich norm: post_attention_layernorm wraps the attention
        # output and the MLP has its own pre/post feedforward norms.
        self._sandwich_norm = hasattr(self.layers[self.source_start], "pre_feedforward_layernorm")

    # === building blocks replayed from the model's own block structure ===

    def _add_attn_out(self, layer, residual: Tensor, attn_out: Tensor) -> Tensor:
        if self._sandwich_norm:
            attn_out = layer.post_attention_layernorm(attn_out)
        # device_map="auto" can put consecutive layers of the band on different
        # GPUs. accelerate realigns the inputs of the modules it wraps, so the
        # block output comes back on that layer's device while the residual is
        # still on the previous one; these adds are ours, so we realign them.
        # No-ops when the whole band sits on one device.
        return residual.to(attn_out.device) + attn_out

    def _mlp_block(self, layer, h: Tensor) -> Tensor:
        residual = h
        if self._sandwich_norm:
            h_normed = layer.pre_feedforward_layernorm(h)
            mlp_out = layer.mlp(h_normed)
            if isinstance(mlp_out, tuple):  # MoE returns (hidden, router_logits)
                mlp_out = mlp_out[0]
            mlp_out = layer.post_feedforward_layernorm(mlp_out)
        else:
            h_normed = layer.post_attention_layernorm(h)
            mlp_out = layer.mlp(h_normed)
            if isinstance(mlp_out, tuple):
                mlp_out = mlp_out[0]
        return residual.to(mlp_out.device) + mlp_out

    def _build_attn_masks(self, S: int, device, dtype) -> Dict[str, Any]:
        """Build per-layer-type masks exactly as transformers does (causal +
        sliding-window causal), broadcast over the factor batch."""
        from transformers.masking_utils import (
            create_causal_mask, create_sliding_window_causal_mask,
        )
        dummy = torch.zeros(1, S, self.hidden_size, device=device, dtype=dtype)
        cache_position = torch.arange(S, device=device)
        mk = dict(config=self.model_config, inputs_embeds=dummy, attention_mask=None,
                  cache_position=cache_position, past_key_values=None)
        masks: Dict[str, Any] = {"full_attention": create_causal_mask(**mk)}
        layer_types = getattr(self.model_config, "layer_types", None)
        if (getattr(self.model_config, "sliding_window", None)
                and layer_types is not None and "sliding_attention" in layer_types):
            masks["sliding_attention"] = create_sliding_window_causal_mask(**mk)
        return masks

    def _mask_for_layer(self, layer_idx: int, masks: Dict[str, Any]):
        layer_types = getattr(self.model_config, "layer_types", None)
        if layer_types is not None:
            return masks.get(layer_types[layer_idx], masks["full_attention"])
        return masks["full_attention"]

    # === forward ===

    def forward_chunk_delta_mean(
        self,
        h: Tensor,
        y: Tensor,
        k_start: int,
        k_end: int,
        target_positions: Union[slice, int],
    ) -> Tensor:
        """Forward a chunk of factors and return the mean activation delta at the
        target positions: (k_chunk, D)."""
        out_chunk = self._forward_chunk(h, k_start, k_end)
        k_chunk = k_end - k_start
        y_expanded = y.unsqueeze(0).expand(k_chunk, -1, -1, -1)
        delta = out_chunk - y_expanded.to(out_chunk.device)
        if isinstance(target_positions, slice):
            delta = delta[:, :, target_positions, :]
        else:
            delta = delta[:, :, target_positions:target_positions + 1, :]
        return delta.mean(dim=(1, 2))

    def _forward_chunk(self, h: Tensor, k_start: int, k_end: int) -> Tensor:
        k_chunk = k_end - k_start
        B, S, D = h.shape
        h_flat = h.unsqueeze(0).expand(k_chunk, -1, -1, -1).reshape(k_chunk * B, S, D)

        position_ids = torch.arange(S, device=h.device).unsqueeze(0).expand(k_chunk * B, -1)
        position_embeddings = self.model.model.rotary_emb(h_flat, position_ids)
        attn_masks = self._build_attn_masks(S, h_flat.device, h_flat.dtype)

        start_layer = min(self.source_start, self.target_layer)
        end_layer = max(self.source_end, self.target_layer)
        for layer_idx in range(start_layer, end_layer + 1):
            layer = self.layers[layer_idx]
            mask = self._mask_for_layer(layer_idx, attn_masks)
            if self.source_start <= layer_idx <= self.source_end:
                h_flat = self._layer_with_lora(
                    layer, h_flat, layer_idx, k_start, k_end, B, k_chunk, S,
                    position_embeddings, mask)
            else:
                h_flat = self._layer_standard(layer, h_flat, position_embeddings, mask)

        return h_flat.reshape(k_chunk, B, S, D)

    def forward_unsteered(self, h: Tensor) -> Tensor:
        """(B, S, D) -> (B, S, D) output at target layer, no LoRA. Used for
        computing/verifying Y and in tests."""
        B, S, D = h.shape
        position_ids = torch.arange(S, device=h.device).unsqueeze(0).expand(B, -1)
        position_embeddings = self.model.model.rotary_emb(h, position_ids)
        attn_masks = self._build_attn_masks(S, h.device, h.dtype)
        for layer_idx in range(min(self.source_start, self.target_layer),
                               max(self.source_end, self.target_layer) + 1):
            layer = self.layers[layer_idx]
            mask = self._mask_for_layer(layer_idx, attn_masks)
            h = self._layer_standard(layer, h, position_embeddings, mask)
        return h

    def _attn_forward(self, layer, h_normed, position_embeddings, mask):
        if hasattr(layer, "self_attn"):
            attn_out, _ = layer.self_attn(
                hidden_states=h_normed,
                position_embeddings=position_embeddings,
                attention_mask=mask,
            )
            return attn_out
        # GatedDeltaNet: a gated recurrence, so it takes neither rotary
        # embeddings nor an attention mask. CPE caches activations one prompt at
        # a time, so the batch is unpadded and there is nothing to mask anyway.
        return layer.linear_attn(hidden_states=h_normed, cache_params=None,
                                 attention_mask=None)

    def _layer_standard(self, layer, h_flat, position_embeddings, mask):
        residual = h_flat
        attn_out = self._attn_forward(layer, layer.input_layernorm(h_flat),
                                      position_embeddings, mask)
        h_flat = self._add_attn_out(layer, residual, attn_out)
        return self._mlp_block(layer, h_flat)

    def _layer_with_lora(self, layer, h_flat, layer_idx, k_start, k_end, B, k_chunk, S,
                         position_embeddings, mask):
        lora_params = self.factors.layer_params(layer_idx, k_start, k_end)
        _, attn = attn_block(layer)
        handles = [
            getattr(attn, resolve_module(attn, module)).register_forward_hook(
                _lora_output_hook(p['A'], p['B'], k_chunk))
            for module, p in lora_params.items()
        ]
        try:
            residual = h_flat
            attn_out = self._attn_forward(layer, layer.input_layernorm(h_flat),
                                          position_embeddings, mask)
        finally:
            for handle in handles:
                handle.remove()
        h_flat = self._add_attn_out(layer, residual, attn_out)
        return self._mlp_block(layer, h_flat)

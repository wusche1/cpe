"""Batched sliced forward: replay layers [min(source), target] of a decoder-only
transformer with K rank-r LoRA perturbations applied in parallel.

Ported from legacy/lora/lora_dct_distributed.py (BatchedSlicedLoRAModel), with the
distributed / pipeline-parallel / CPU-offload machinery removed. The forward replays
the model's OWN attention interface, rotary function, and mask construction so it is
numerically consistent with a normal model forward.
"""

import importlib
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


def apply_batched_lora(x_4d: Tensor, A_batch: Tensor, B_batch: Tensor) -> Tensor:
    """Apply K LoRA deltas in parallel.

    x_4d: (K, B, S, d_in); A_batch: (K, r, d_in); B_batch: (K, d_out, r)
    Returns (K, B, S, d_out).
    """
    intermediate = torch.einsum('kbsi,kri->kbsr', x_4d, A_batch)
    return torch.einsum('kbsr,kor->kbso', intermediate, B_batch)


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
        return residual + attn_out

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
        return residual + mlp_out

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

    def _run_attn_interface(self, attn, q, k, v, mask):
        """Run the model's own attention interface (GQA repeat, sinks, sliding
        window) on projected q/k/v. Returns (batch, S, hidden)."""
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        modeling_mod = importlib.import_module(type(attn).__module__)
        default_eager = getattr(modeling_mod, "eager_attention_forward")
        impl = getattr(self.model_config, "_attn_implementation", "eager")
        fn = ALL_ATTENTION_FUNCTIONS.get_interface(impl, default_eager)
        scaling = getattr(attn, "scaling", self.head_dim ** -0.5)
        attn_out, _ = fn(
            attn, q, k, v, mask,
            dropout=0.0,
            scaling=scaling,
            sliding_window=getattr(attn, "sliding_window", None),
            s_aux=getattr(attn, "sinks", None),
        )
        return attn_out.reshape(attn_out.shape[0], attn_out.shape[1], -1)

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
        delta = out_chunk - y_expanded
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

    def _layer_standard(self, layer, h_flat, position_embeddings, mask):
        residual = h_flat
        h_normed = layer.input_layernorm(h_flat)
        attn_out, _ = layer.self_attn(
            hidden_states=h_normed,
            position_embeddings=position_embeddings,
            attention_mask=mask,
        )
        h_flat = self._add_attn_out(layer, residual, attn_out)
        return self._mlp_block(layer, h_flat)

    def _layer_with_lora(self, layer, h_flat, layer_idx, k_start, k_end, B, k_chunk, S,
                         position_embeddings, mask):
        lora_params = self.factors.layer_params(layer_idx, k_start, k_end)
        residual = h_flat
        h_normed = layer.input_layernorm(h_flat)
        attn_out = self._attention_with_lora(
            layer.self_attn, h_normed, lora_params, B, k_chunk, S,
            position_embeddings, mask)
        h_flat = self._add_attn_out(layer, residual, attn_out)
        return self._mlp_block(layer, h_flat)

    def _attention_with_lora(self, attn, h_normed, lora_params, B, k_chunk, S,
                             position_embeddings, mask):
        D = h_normed.shape[-1]
        h_4d = h_normed.reshape(k_chunk, B, S, D)

        def proj(name):
            base = getattr(attn, name)(h_normed)
            if name in lora_params:
                delta = apply_batched_lora(h_4d, lora_params[name]['A'], lora_params[name]['B'])
                base = (base.reshape(k_chunk, B, S, -1) + delta).reshape(k_chunk * B, S, -1)
            return base

        q, k, v = proj('q_proj'), proj('k_proj'), proj('v_proj')

        if hasattr(attn, 'q_norm'):
            q = attn.q_norm(q.view(*q.shape[:-1], -1, self.head_dim)).view(q.shape)
        if hasattr(attn, 'k_norm'):
            k = attn.k_norm(k.view(*k.shape[:-1], -1, self.head_dim)).view(k.shape)

        q = q.view(k_chunk * B, S, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(k_chunk * B, S, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(k_chunk * B, S, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # The model's own rotary function (conventions differ across families).
        apply_rotary = importlib.import_module(type(attn).__module__).apply_rotary_pos_emb
        cos, sin = position_embeddings
        q, k = apply_rotary(q, k, cos, sin)

        attn_output = self._run_attn_interface(attn, q, k, v, mask)

        o_base = attn.o_proj(attn_output)
        if 'o_proj' in lora_params:
            delta_o = apply_batched_lora(
                attn_output.reshape(k_chunk, B, S, -1),
                lora_params['o_proj']['A'], lora_params['o_proj']['B'])
            o_base = (o_base.reshape(k_chunk, B, S, -1) + delta_o).reshape(k_chunk * B, S, -1)
        return o_base

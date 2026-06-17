"""
Distributed LoRA DCT (Deep Causal Transcoding) Implementation

This module implements LoRA-based DCT training where instead of learning steering vectors,
we learn low-rank adapter matrices that maximize the change in target layer activations.

Key features:
- Batched parallel processing of multiple LoRA factors within each GPU
- Distributed training across multiple GPUs with factor parallelism
- Soft orthogonalization over flattened LoRA parameters
- PEFT-compatible export format
"""

import functools
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Callable, Union
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from tqdm import tqdm


# =============================================================================
# Utility Functions (from original DCT)
# =============================================================================

def rgetattr(obj, path):
    """Recursive getattr for nested attributes like 'model.layers'."""
    return functools.reduce(getattr, path.split("."), obj)


def rhasattr(obj, path):
    """Recursive hasattr for nested attributes."""
    try:
        rgetattr(obj, path)
        return True
    except (AttributeError, TypeError):
        return False


class StreamingAverage:
    """
    Maintains a streaming average of tensors.
    Handles variable batch sizes and arbitrary tensor dimensions.
    """
    def __init__(self):
        self.count = 0
        self.mean = None
    
    def update(self, batch: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Updates the streaming average with a new batch of data.
        
        Args:
            batch: Tensor of shape (batch_size, dim1, ..., dimk)
            mask: Optional tensor for masked averaging
        
        Returns:
            Current mean after incorporating the new batch
        """
        if mask is not None:
            masked_batch = batch * mask
            batch_sum = masked_batch.sum(dim=0)
            mask_sum = mask.sum(dim=0)
            batch_mean = batch_sum / mask_sum.clamp(min=1.0)
            batch_size = mask.sum().item()
        else:
            batch_size = batch.size(0)
            batch_mean = batch.mean(dim=0)
        
        if self.mean is None:
            self.mean = batch_mean
            self.count = batch_size
            return self.mean
        
        new_count = self.count + batch_size
        self.mean = self.mean + (batch_mean - self.mean) * (batch_size / new_count)
        self.count = new_count
        
        return self.mean
    
    def get_mean(self) -> torch.Tensor:
        """Returns the current mean."""
        if self.mean is None:
            raise ValueError("No data has been processed yet")
        return self.mean
    
    def reset(self):
        """Resets the streaming average."""
        self.count = 0
        self.mean = None


def soft_ortho(V, num_iterations=1, temperature=1.0, logit_bias=None):
    """
    Performs soft orthogonalization of vectors in V using vectorized operations.
    
    Args:
        V: Tensor of shape (d, n) where d is the dimension and n is the number of vectors
        num_iterations: Number of times to repeat the process
        temperature: Temperature parameter for the softmax (higher = softer)
        logit_bias: Optional tensor of shape (n) to add as bias in the softmax
        
    Returns:
        Updated tensor of the same shape
    """
    d, n = V.shape
    
    if logit_bias is None:
        logit_bias = torch.zeros(n, device=V.device)
    
    # Normalize columns initially
    V = F.normalize(V, p=2, dim=0)
    
    # Create masks for excluding self-interactions
    mask = torch.ones(n, n, device=V.device) - torch.eye(n, device=V.device)
    
    for _ in range(num_iterations):
        # Compute all pairwise dot products
        dot_products = torch.matmul(V.T, V) / temperature
        
        # Add logit bias
        dot_products = dot_products + logit_bias.unsqueeze(0)
        
        # Set diagonal to -inf for softmax
        dot_products = dot_products.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to each row
        weights = F.softmax(dot_products, dim=1)
        
        # Compute weighted sums
        weighted_sums = torch.matmul(V, weights.T)
        
        # Update vectors
        V_new = V - weighted_sums
        V_new = F.normalize(V_new, p=2, dim=0)
        V = V_new
    
    return V


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class LoRAFactorConfig:
    """Configuration for LoRA DCT training."""
    
    # Layer configuration
    source_layers: Tuple[int, int]  # (start, end) inclusive - layers to apply LoRA
    target_layer: int  # Layer to measure activation changes
    
    # LoRA configuration
    target_modules: List[str] = field(default_factory=lambda: ["o_proj"])
    lora_rank: int = 1
    lora_alpha: float = 1.0  # For PEFT compatibility (scaling = alpha / rank)
    
    # Normalization
    norm_value: float = 1.0  # Target norm for each column of A.T and B
    
    # Target positions for measuring delta
    target_position_indices: Union[slice, int] = field(default_factory=lambda: slice(-3, None))
    
    def __post_init__(self):
        if isinstance(self.target_modules, str):
            self.target_modules = [self.target_modules]


# =============================================================================
# Batched LoRA Application
# =============================================================================

def apply_batched_lora(x_4d: Tensor, A_batch: Tensor, B_batch: Tensor) -> Tensor:
    """
    Apply K different LoRA adaptations in parallel using einsum.
    
    Args:
        x_4d: (K, B, S, d_in) - input activations, replicated for each factor
        A_batch: (K, r, d_in) - per-factor A matrices
        B_batch: (K, d_out, r) - per-factor B matrices
        
    Returns:
        delta: (K, B, S, d_out) - LoRA contributions for each factor
    """
    # x @ A.T: (K, B, S, d_in) @ (K, d_in, r) -> (K, B, S, r)
    intermediate = torch.einsum('kbsi,kri->kbsr', x_4d, A_batch)
    
    # intermediate @ B.T: (K, B, S, r) @ (K, r, d_out) -> (K, B, S, d_out)
    delta = torch.einsum('kbsr,kor->kbso', intermediate, B_batch)
    
    return delta


# =============================================================================
# LoRA Factor Set
# =============================================================================

class LoRAFactorSet(nn.Module):
    """
    Stores K LoRA factors with batched parameter tensors.
    
    For efficient parallel application, parameters are stored as:
        A[layer_idx][module_name]: (K, r, d_in)
        B[layer_idx][module_name]: (K, d_out, r)
    
    This allows vectorized application of all K LoRA factors simultaneously.
    """
    
    def __init__(
        self,
        num_factors: int,
        config: LoRAFactorConfig,
        model: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        offload_to_cpu: bool = True
    ):
        super().__init__()
        self.num_factors = num_factors
        self.config = config
        self.rank = config.lora_rank
        self.dtype = dtype

        # CPU offloading: store factors on CPU, load to GPU on-demand
        self.offload_to_cpu = offload_to_cpu
        self.gpu_device = device
        self.device = torch.device('cpu') if offload_to_cpu else device
        
        # Inspect model to get dimensions for each target module
        self.module_dims = self._get_module_dims(model)
        
        # Initialize parameter dictionaries
        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        
        # Track parameter layout for flattening/unflattening
        self._param_layout = []
        
        for layer_idx in range(config.source_layers[0], config.source_layers[1] + 1):
            for module_name in config.target_modules:
                d_in, d_out = self.module_dims[layer_idx][module_name]
                key = f"layer{layer_idx}_{module_name}"
                
                # A: (K, r, d_in), B: (K, d_out, r)
                self.A[key] = nn.Parameter(
                    torch.zeros(num_factors, self.rank, d_in, device=device, dtype=dtype)
                )
                self.B[key] = nn.Parameter(
                    torch.zeros(num_factors, d_out, self.rank, device=device, dtype=dtype)
                )
                
                # Track layout
                self._param_layout.append({
                    'key': key,
                    'A_shape': (num_factors, self.rank, d_in),
                    'B_shape': (num_factors, d_out, self.rank),
                    'A_numel': self.rank * d_in,
                    'B_numel': d_out * self.rank,
                })
        
        # Compute total flattened dimension
        self._flat_dim = sum(p['A_numel'] + p['B_numel'] for p in self._param_layout)
    
    def _get_module_dims(self, model: nn.Module) -> Dict[int, Dict[str, Tuple[int, int]]]:
        """
        Inspect model to get input/output dimensions for each target module.
        
        Returns:
            Dict mapping layer_idx -> module_name -> (d_in, d_out)
        """
        dims = {}
        layers = model.model.layers
        
        for layer_idx in range(self.config.source_layers[0], self.config.source_layers[1] + 1):
            layer = layers[layer_idx]
            dims[layer_idx] = {}
            
            for module_name in self.config.target_modules:
                # Get the linear layer
                if hasattr(layer.self_attn, module_name):
                    linear = getattr(layer.self_attn, module_name)
                    d_in = linear.in_features
                    d_out = linear.out_features
                    dims[layer_idx][module_name] = (d_in, d_out)
                else:
                    raise ValueError(f"Module {module_name} not found in layer {layer_idx}")
        
        return dims
    
    def get_layer_params(self, layer_idx: int, k_start: int = 0, k_end: int = None,
                         target_device=None) -> Dict[str, Dict[str, Tensor]]:
        """
        Get A, B parameters for all target modules at a specific layer.

        Args:
            layer_idx: The layer index
            k_start: Start index for factor slicing
            k_end: End index for factor slicing (None = all)
            target_device: Device to place params on (None = self.gpu_device).
                Used by pipeline-parallel models where layers are on different devices.

        Returns:
            Dict mapping module_name -> {'A': tensor, 'B': tensor}
        """
        if k_end is None:
            k_end = self.num_factors
        if target_device is None:
            target_device = self.gpu_device

        params = {}
        for module_name in self.config.target_modules:
            key = f"layer{layer_idx}_{module_name}"
            A_slice = self.A[key][k_start:k_end]
            B_slice = self.B[key][k_start:k_end]

            # Move to target GPU
            if self.offload_to_cpu or A_slice.device != target_device:
                A_slice = A_slice.to(target_device, non_blocking=True)
                B_slice = B_slice.to(target_device, non_blocking=True)

            params[module_name] = {
                'A': A_slice,
                'B': B_slice,
            }
        return params
    
    def get_flattened_all(self) -> Tensor:
        """
        Return all parameters as (K, D) where D is total param count per factor.
        Used for soft orthogonalization.
        """
        vecs = []
        for layout in self._param_layout:
            key = layout['key']
            # A: (K, r, d_in) -> (K, r*d_in)
            vecs.append(self.A[key].reshape(self.num_factors, -1))
            # B: (K, d_out, r) -> (K, d_out*r)
            vecs.append(self.B[key].reshape(self.num_factors, -1))
        return torch.cat(vecs, dim=1)  # (K, D_total)
    
    def set_from_flattened(self, flattened: Tensor):
        """Inverse of get_flattened_all - set parameters from flattened tensor."""
        offset = 0
        for layout in self._param_layout:
            key = layout['key']
            A_numel = layout['A_numel']
            B_numel = layout['B_numel']
            A_shape = layout['A_shape']
            B_shape = layout['B_shape']
            
            self.A[key].data = flattened[:, offset:offset+A_numel].reshape(A_shape)
            offset += A_numel
            
            self.B[key].data = flattened[:, offset:offset+B_numel].reshape(B_shape)
            offset += B_numel
    
    def get_grad_flattened(self) -> Tensor:
        """Get gradients in same format as get_flattened_all."""
        grads = []
        for layout in self._param_layout:
            key = layout['key']
            if self.A[key].grad is None:
                grads.append(torch.zeros(self.num_factors, layout['A_numel'], 
                                        device=self.device, dtype=self.dtype))
            else:
                grads.append(self.A[key].grad.reshape(self.num_factors, -1))
            
            if self.B[key].grad is None:
                grads.append(torch.zeros(self.num_factors, layout['B_numel'],
                                        device=self.device, dtype=self.dtype))
            else:
                grads.append(self.B[key].grad.reshape(self.num_factors, -1))
        return torch.cat(grads, dim=1)
    
    def zero_grad(self):
        """Zero all gradients."""
        for key in self.A.keys():
            if self.A[key].grad is not None:
                self.A[key].grad.zero_()
            if self.B[key].grad is not None:
                self.B[key].grad.zero_()
    
    def normalize_columns(self, norm_value: float):
        """
        Normalize each rank-column of A and B independently.
        
        For A of shape (K, r, d_in): normalize each of the r rows to have norm = norm_value
        For B of shape (K, d_out, r): normalize each of the r columns to have norm = norm_value
        """
        for key in self.A.keys():
            A = self.A[key]  # (K, r, d_in)
            B = self.B[key]  # (K, d_out, r)
            
            # Normalize rows of A: norm along dim=2
            A_norms = A.norm(dim=2, keepdim=True).clamp(min=1e-8)
            self.A[key].data = A * (norm_value / A_norms)
            
            # Normalize columns of B: norm along dim=1
            B_norms = B.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self.B[key].data = B * (norm_value / B_norms)
    
    def init_random(self, scale: float = 0.01):
        """Random initialization with small scale (standard LoRA: A random, B zeros)."""
        for key in self.A.keys():
            nn.init.normal_(self.A[key], std=scale)
            nn.init.zeros_(self.B[key])
    
    def init_kaiming(self):
        """Kaiming initialization for A, zeros for B (matches PEFT default)."""
        for key in self.A.keys():
            # Kaiming uniform for A
            nn.init.kaiming_uniform_(self.A[key].view(-1, self.A[key].shape[-1]), a=math.sqrt(5))
            nn.init.zeros_(self.B[key])
    
    @property
    def flat_dim(self) -> int:
        """Total flattened dimension per factor."""
        return self._flat_dim
    
    def to_peft_state_dict(self, factor_idx: int, adapter_name: str = "default") -> Dict[str, Tensor]:
        """
        Export a single factor to PEFT-compatible state dict.
        
        The returned dict can be used with PEFT's load_state_dict.
        """
        state = {}
        
        for layer_idx in range(self.config.source_layers[0], self.config.source_layers[1] + 1):
            for module_name in self.config.target_modules:
                key = f"layer{layer_idx}_{module_name}"
                
                A = self.A[key][factor_idx].detach()  # (r, d_in)
                B = self.B[key][factor_idx].detach()  # (d_out, r)
                
                # PEFT naming convention
                prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{module_name}"
                state[f"{prefix}.lora_A.{adapter_name}.weight"] = A.cpu()
                state[f"{prefix}.lora_B.{adapter_name}.weight"] = B.cpu()
        
        return state


# =============================================================================
# Batched Sliced LoRA Model
# =============================================================================

class BatchedSlicedLoRAModel(nn.Module):
    """
    Runs forward pass from source_layers through target_layer with batched LoRA application.
    
    Processes all K factors in parallel by expanding the batch dimension and
    using einsum for efficient LoRA application.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        config: LoRAFactorConfig, 
        lora_factors: LoRAFactorSet
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.lora_factors = lora_factors
        
        # Get reference to layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.layers = model.model.layers
        elif hasattr(model, 'layers'):
            self.layers = model.layers
        else:
            raise ValueError("Cannot find layers in model")
        
        # Layer indices
        self.source_start = config.source_layers[0]
        self.source_end = config.source_layers[1]
        self.target_layer = config.target_layer

        # Validate that required layers exist (may be None if model was trimmed)
        start_layer = min(self.source_start, self.target_layer)
        end_layer = max(self.source_end, self.target_layer)
        for layer_idx in range(start_layer, end_layer + 1):
            if self.layers[layer_idx] is None:
                raise ValueError(f"Layer {layer_idx} is None but required for training "
                               f"(source_layers={config.source_layers}, target_layer={config.target_layer})")

        # Get model config for attention computation
        self.model_config = model.config
        self.hidden_size = model.config.hidden_size
        self.num_attention_heads = model.config.num_attention_heads
        self.num_key_value_heads = model.config.num_key_value_heads
        self.head_dim = getattr(model.config, 'head_dim', 
                                model.config.hidden_size // model.config.num_attention_heads)
        
        # Device
        self.device = next(model.parameters()).device

        # Gemma-2/3 use a "sandwich" norm: post_attention_layernorm wraps the
        # ATTENTION OUTPUT (not the pre-MLP hidden state), and the MLP block is
        # wrapped by pre_feedforward_layernorm / post_feedforward_layernorm.
        # Standard models (llama/qwen/gpt-oss) instead apply post_attention_layernorm
        # as the pre-MLP norm and have no feedforward norms. Detect once so the
        # sliced forward replays the exact block structure.
        self._sandwich_norm = hasattr(self.layers[self.source_start], "pre_feedforward_layernorm")

        # Validate Flash Attention availability at startup
        self._validate_flash_attention()

    def _add_attn_out(self, layer, residual: Tensor, attn_out: Tensor) -> Tensor:
        """Add the attention output to the residual, applying Gemma-2/3's
        post_attention_layernorm to the attention output first when present."""
        if self._sandwich_norm:
            attn_out = layer.post_attention_layernorm(attn_out)
        return residual + attn_out

    def _mlp_block(self, layer, h_flat: Tensor) -> Tensor:
        """Run the MLP residual block. Standard: post_attention_layernorm -> mlp.
        Gemma-2/3 sandwich: pre_feedforward_layernorm -> mlp -> post_feedforward_layernorm."""
        residual = h_flat
        if self._sandwich_norm:
            h_normed = layer.pre_feedforward_layernorm(h_flat)
            mlp_out = layer.mlp(h_normed)
            if isinstance(mlp_out, tuple):  # MoE returns (hidden, router_logits)
                mlp_out = mlp_out[0]
            mlp_out = layer.post_feedforward_layernorm(mlp_out)
        else:
            h_normed = layer.post_attention_layernorm(h_flat)
            mlp_out = layer.mlp(h_normed)
            if isinstance(mlp_out, tuple):
                mlp_out = mlp_out[0]
        return residual + mlp_out

    def _validate_flash_attention(self):
        """Validate that Flash Attention is available and will work."""
        if not torch.cuda.is_available():
            raise RuntimeError("Flash Attention requires CUDA")

        # Test Flash Attention with a small tensor
        dtype = next(self.model.parameters()).dtype

        test_q = torch.randn(1, 1, 4, 64, device=self.device, dtype=dtype)
        test_k = torch.randn(1, 1, 4, 64, device=self.device, dtype=dtype)
        test_v = torch.randn(1, 1, 4, 64, device=self.device, dtype=dtype)

        try:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_mem_efficient=False,
                enable_math=False,
            ):
                _ = F.scaled_dot_product_attention(test_q, test_k, test_v, is_causal=True)
        except RuntimeError as e:
            raise RuntimeError(
                f"Flash Attention unavailable. Requirements: "
                f"GPU compute capability >= 8.0 (Ampere+), PyTorch >= 2.0, bf16/fp16 dtype. "
                f"Error: {e}"
            )

    def forward(
        self,
        h: Tensor,
        factor_batch_size: int = 16
    ) -> Tensor:
        """
        Forward pass with all LoRA factors applied in parallel.

        Args:
            h: (B, S, D) - hidden states at input to first layer in range
            factor_batch_size: Chunk size for processing factors to manage memory

        Returns:
            (K, B, S, D) - output at target_layer for each factor
        """
        K = self.lora_factors.num_factors
        B, S, D = h.shape

        outputs = []
        for k_start in range(0, K, factor_batch_size):
            k_end = min(k_start + factor_batch_size, K)
            out_chunk = self._forward_chunk(h, k_start, k_end)
            outputs.append(out_chunk)

        return torch.cat(outputs, dim=0)  # (K, B, S, D)

    def forward_chunk_delta_mean(
        self,
        h: Tensor,
        y: Tensor,
        k_start: int,
        k_end: int,
        target_position_indices: Union[slice, int]
    ) -> Tensor:
        """
        Forward pass for a chunk of factors and compute delta_mean.

        This avoids materializing the full (K, B, S, D) tensor by computing
        delta_mean directly for each chunk.

        Args:
            h: (B, S, D) - input hidden states
            y: (B, S, D) - target hidden states
            k_start: Start index for factor chunk
            k_end: End index for factor chunk
            target_position_indices: Slice or int for target positions

        Returns:
            (k_chunk, D) - mean delta for each factor in chunk
        """
        # Forward pass for this chunk: (k_chunk, B, S, D)
        out_chunk = self._forward_chunk(h, k_start, k_end)
        k_chunk = k_end - k_start

        # Expand y: (B, S, D) -> (k_chunk, B, S, D)
        # Move y to out_chunk's device (for pipeline-parallel models, the output
        # may be on a different device than where y was originally computed)
        y_expanded = y.to(out_chunk.device).unsqueeze(0).expand(k_chunk, -1, -1, -1)

        # Compute delta
        delta = out_chunk - y_expanded

        # Slice target positions
        if isinstance(target_position_indices, slice):
            delta = delta[:, :, target_position_indices, :]
        else:
            delta = delta[:, :, target_position_indices:target_position_indices+1, :]

        # Mean over batch and positions: (k_chunk, D)
        delta_mean = delta.mean(dim=(1, 2))

        return delta_mean

    def _build_attn_masks(self, S: int, device, dtype) -> Dict[str, Any]:
        """Build per-layer-type attention masks matching the model's OWN masking.

        Replays the exact masks transformers builds (full causal + sliding-window
        causal), in the format expected by the loaded attn implementation, so the
        sliced forward attends identically to a normal model forward. Built with a
        dummy batch=1 (no padding) and broadcast over the factor batch.
        """
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
        """Select the mask for a layer by its layer_type (default full causal)."""
        layer_types = getattr(self.model_config, "layer_types", None)
        if layer_types is not None:
            return masks.get(layer_types[layer_idx], masks["full_attention"])
        return masks["full_attention"]

    def _run_attn_interface(self, attn, q, k, v, mask):
        """Run the model's OWN attention interface (handles GQA repeat, attention
        sinks, sliding window) on already-projected q/k/v. Returns (batch, S, hidden).

        q: (batch, n_heads, S, hd); k,v: (batch, n_kv, S, hd). KV is NOT pre-repeated
        (the interface repeats it). s_aux/sliding_window/scaling are read from the
        attn module so this is exact for gpt-oss (sinks), qwen3, llama, etc.
        """
        import importlib
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

    def _forward_chunk(
        self,
        h: Tensor,
        k_start: int,
        k_end: int
    ) -> Tensor:
        """Forward pass for a chunk of factors."""
        k_chunk = k_end - k_start
        B, S, D = h.shape

        # Expand: (B, S, D) -> (k_chunk, B, S, D) -> (k_chunk*B, S, D)
        h_exp = h.unsqueeze(0).expand(k_chunk, -1, -1, -1)
        h_flat = h_exp.reshape(k_chunk * B, S, D)

        # We need position embeddings for attention
        # Get them from the model's rotary embedding
        position_ids = torch.arange(S, device=h.device).unsqueeze(0).expand(k_chunk * B, -1)

        # Get position embeddings (cos, sin)
        if hasattr(self.model.model, 'rotary_emb'):
            rotary = self.model.model.rotary_emb
            # For pipeline-parallel models, rotary_emb may be on a different device
            rotary_device = next(rotary.parameters(), None)
            if rotary_device is not None:
                rotary_device = rotary_device.device
            else:
                rotary_device = h_flat.device
            position_embeddings = rotary(
                h_flat.to(rotary_device), position_ids.to(rotary_device),
            )
            # Move back to h_flat's device (will be re-migrated per-layer if needed)
            position_embeddings = (
                position_embeddings[0].to(h.device),
                position_embeddings[1].to(h.device),
            )
        else:
            # Fallback - compute simple rotary embeddings
            position_embeddings = self._compute_rotary_embeddings(h_flat, position_ids)

        # Build per-layer-type attention masks once (causal + sliding window),
        # matching the model's own masking so the sliced forward is exact.
        attn_masks = self._build_attn_masks(S, h_flat.device, h_flat.dtype)

        # Process layers from min(source_layers) to target_layer
        start_layer = min(self.source_start, self.target_layer)
        end_layer = max(self.source_end, self.target_layer)

        for layer_idx in range(start_layer, end_layer + 1):
            layer = self.layers[layer_idx]

            # Cross-device migration for pipeline-parallel models (training_tp_size > 1)
            layer_device = next(layer.parameters()).device
            if h_flat.device != layer_device:
                h_flat = h_flat.to(layer_device)
                position_embeddings = (
                    position_embeddings[0].to(layer_device),
                    position_embeddings[1].to(layer_device),
                )
                attn_masks = {kt: (m.to(layer_device) if torch.is_tensor(m) else m)
                              for kt, m in attn_masks.items()}

            mask = self._mask_for_layer(layer_idx, attn_masks)

            if self.source_start <= layer_idx <= self.source_end:
                # Apply LoRA to this layer
                h_flat = self._forward_layer_with_lora(
                    layer, h_flat, layer_idx, k_start, k_end, B, k_chunk, S,
                    position_embeddings, mask
                )
            else:
                # Standard forward (no LoRA)
                h_flat = self._forward_layer_standard(
                    layer, h_flat, k_chunk, B, S, position_embeddings, mask
                )

        # Reshape back: (k_chunk*B, S, D) -> (k_chunk, B, S, D)
        return h_flat.reshape(k_chunk, B, S, D)
    
    def _compute_rotary_embeddings(self, x: Tensor, position_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute rotary embeddings when model's rotary_emb isn't accessible."""
        # Simple implementation - may need adjustment for specific models
        seq_len = position_ids.shape[1]
        base = getattr(self.model_config, 'rope_theta', 10000.0)
        dim = self.head_dim
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
        t = position_ids.float()
        freqs = torch.einsum('bi,j->bij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()
        sin = emb.sin()
        
        return cos.to(x.dtype), sin.to(x.dtype)
    
    def _forward_layer_standard(
        self,
        layer,
        h_flat: Tensor,
        k_chunk: int,
        B: int,
        S: int,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask=None,
    ) -> Tensor:
        """Standard forward through a layer without LoRA."""
        residual = h_flat
        h_normed = layer.input_layernorm(h_flat)

        # Attention (pass the model's own mask so causal + sliding window apply)
        attn_out, _ = layer.self_attn(
            hidden_states=h_normed,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        h_flat = self._add_attn_out(layer, residual, attn_out)

        # MLP residual block (standard or Gemma sandwich norm)
        h_flat = self._mlp_block(layer, h_flat)

        return h_flat

    def _forward_layer_with_lora(
        self,
        layer,
        h_flat: Tensor,
        layer_idx: int,
        k_start: int,
        k_end: int,
        B: int,
        k_chunk: int,
        S: int,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask=None,
    ) -> Tensor:
        """
        Forward through a decoder layer with LoRA applied to attention projections.

        h_flat: (k_chunk*B, S, D)
        """
        D = h_flat.shape[-1]

        # Get LoRA params for this layer (on same device as the layer)
        lora_params = self.lora_factors.get_layer_params(
            layer_idx, k_start, k_end, target_device=h_flat.device,
        )

        # === Input LayerNorm ===
        residual = h_flat
        h_normed = layer.input_layernorm(h_flat)

        # === Attention with LoRA ===
        attn_out = self._attention_with_lora(
            layer.self_attn, h_normed, lora_params, B, k_chunk, S, position_embeddings,
            attention_mask
        )
        
        h_flat = self._add_attn_out(layer, residual, attn_out)

        # === MLP (no LoRA for now) — standard or Gemma sandwich norm ===
        h_flat = self._mlp_block(layer, h_flat)

        return h_flat
    
    def _attention_with_lora(
        self, 
        attn, 
        h_normed: Tensor, 
        lora_params: Dict[str, Dict[str, Tensor]],
        B: int,
        k_chunk: int,
        S: int,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask=None,
    ) -> Tensor:
        """
        Compute attention with LoRA applied to Q, K, V, O projections.

        h_normed: (k_chunk*B, S, D)
        """
        D = h_normed.shape[-1]
        
        # Reshape for LoRA: (k_chunk*B, S, D) -> (k_chunk, B, S, D)
        h_4d = h_normed.reshape(k_chunk, B, S, D)
        
        # === Q projection with LoRA ===
        q_base = attn.q_proj(h_normed)  # (k_chunk*B, S, d_q)
        if 'q_proj' in lora_params:
            delta_q = apply_batched_lora(h_4d, lora_params['q_proj']['A'], lora_params['q_proj']['B'])
            q_base_4d = q_base.reshape(k_chunk, B, S, -1)
            q = (q_base_4d + delta_q).reshape(k_chunk * B, S, -1)
        else:
            q = q_base
        
        # === K projection with LoRA ===
        k_base = attn.k_proj(h_normed)
        if 'k_proj' in lora_params:
            delta_k = apply_batched_lora(h_4d, lora_params['k_proj']['A'], lora_params['k_proj']['B'])
            k_base_4d = k_base.reshape(k_chunk, B, S, -1)
            k = (k_base_4d + delta_k).reshape(k_chunk * B, S, -1)
        else:
            k = k_base
        
        # === V projection with LoRA ===
        v_base = attn.v_proj(h_normed)
        if 'v_proj' in lora_params:
            delta_v = apply_batched_lora(h_4d, lora_params['v_proj']['A'], lora_params['v_proj']['B'])
            v_base_4d = v_base.reshape(k_chunk, B, S, -1)
            v = (v_base_4d + delta_v).reshape(k_chunk * B, S, -1)
        else:
            v = v_base
        
        # === Apply Q/K norms if present (Qwen3Moe has these) ===
        if hasattr(attn, 'q_norm'):
            q = attn.q_norm(q.view(*q.shape[:-1], -1, self.head_dim)).view(q.shape)
        if hasattr(attn, 'k_norm'):
            k = attn.k_norm(k.view(*k.shape[:-1], -1, self.head_dim)).view(k.shape)
        
        # === Reshape for attention: (batch, heads, seq, head_dim) ===
        q = q.view(k_chunk * B, S, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(k_chunk * B, S, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(k_chunk * B, S, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # === Apply rotary position embedding (use the model's OWN function so the
        # convention matches exactly — gpt-oss applies half-dim cos/sin over the full
        # head_dim, qwen3 uses doubled cos/sin + rotate_half) ===
        import importlib
        apply_rotary = importlib.import_module(type(attn).__module__).apply_rotary_pos_emb
        cos, sin = position_embeddings
        q, k = apply_rotary(q, k, cos, sin)
        
        # === Attention via the model's OWN interface ===
        # Handles GQA kv-repeat, attention sinks (gpt-oss), sliding window, and the
        # passed causal/sliding mask — exact for gpt-oss/qwen3/llama. q/k/v are NOT
        # pre-repeated; the interface repeats kv internally.
        # q: (k_chunk*B, n_heads, S, hd); k,v: (k_chunk*B, n_kv, S, hd)
        attn_output = self._run_attn_interface(attn, q, k, v, attention_mask)
        # -> (k_chunk*B, S, hidden)
        
        # === O projection with LoRA ===
        o_base = attn.o_proj(attn_output)
        if 'o_proj' in lora_params:
            attn_output_4d = attn_output.reshape(k_chunk, B, S, -1)
            delta_o = apply_batched_lora(attn_output_4d, lora_params['o_proj']['A'], lora_params['o_proj']['B'])
            o_base_4d = o_base.reshape(k_chunk, B, S, -1)
            o = (o_base_4d + delta_o).reshape(k_chunk * B, S, -1)
        else:
            o = o_base
        
        return o
    
    def _apply_rotary_pos_emb(
        self, 
        q: Tensor, 
        k: Tensor, 
        cos: Tensor, 
        sin: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Apply rotary position embeddings to Q and K.

        Handles partial rotary (rotary_dim < head_dim) by only rotating
        the first rotary_dim elements and passing the rest through unchanged.
        """
        # cos, sin: (batch, seq, rotary_dim) -> need (batch, 1, seq, rotary_dim)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        rotary_dim = cos.shape[-1]

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        # Partial rotary: only rotate first rotary_dim dimensions
        if rotary_dim < q.shape[-1]:
            q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
            k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
            q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
            k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)
            q_embed = torch.cat([q_rot, q_pass], dim=-1)
            k_embed = torch.cat([k_rot, k_pass], dim=-1)
        else:
            q_embed = (q * cos) + (rotate_half(q) * sin)
            k_embed = (k * cos) + (rotate_half(k) * sin)

        return q_embed, k_embed
    
    def forward_unsteered(
        self, 
        h: Tensor, 
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass without any LoRA (unsteered baseline).
        
        Args:
            h: (B, S, D) - hidden states
            
        Returns:
            (B, S, D) - output at target_layer
        """
        B, S, D = h.shape
        h_flat = h
        
        # Get position embeddings
        position_ids = torch.arange(S, device=h.device).unsqueeze(0).expand(B, -1)
        if hasattr(self.model.model, 'rotary_emb'):
            position_embeddings = self.model.model.rotary_emb(h_flat, position_ids)
        else:
            position_embeddings = self._compute_rotary_embeddings(h_flat, position_ids)
        
        # Build per-layer-type attention masks (causal + sliding window).
        attn_masks = self._build_attn_masks(S, h_flat.device, h_flat.dtype)

        # Process layers
        start_layer = min(self.source_start, self.target_layer)
        end_layer = max(self.source_end, self.target_layer)

        for layer_idx in range(start_layer, end_layer + 1):
            layer = self.layers[layer_idx]
            mask = self._mask_for_layer(layer_idx, attn_masks)

            residual = h_flat
            h_normed = layer.input_layernorm(h_flat)

            attn_out, _ = layer.self_attn(
                hidden_states=h_normed,
                position_embeddings=position_embeddings,
                attention_mask=mask,
            )
            h_flat = self._add_attn_out(layer, residual, attn_out)

            # MLP residual block (standard or Gemma sandwich norm)
            h_flat = self._mlp_block(layer, h_flat)

        return h_flat


# =============================================================================
# Distributed Soft Orthogonalization for LoRA
# =============================================================================

def distributed_soft_ortho_lora(
    theta_local: Tensor,
    G_theta_norms_local: Tensor,
    rank: int,
    world_size: int,
    num_iterations: int = 1,
    temperature: float = 1.0,
    process_group=None,
) -> Tensor:
    """
    Distributed soft orthogonalization over flattened LoRA params.

    Args:
        theta_local: (local_K, D) flattened LoRA params for local factors
        G_theta_norms_local: (local_K,) gradient norms for logit bias
        rank: Current process rank
        world_size: Total number of processes
        num_iterations: Soft ortho iterations
        temperature: Softmax temperature
        process_group: Optional NCCL process group (None = default global)

    Returns:
        theta_ortho_local: (local_K, D) orthogonalized local params
    """
    device = theta_local.device
    local_K, D = theta_local.shape

    # All-gather theta from all ranks
    theta_list = [torch.zeros_like(theta_local) for _ in range(world_size)]
    dist.all_gather(theta_list, theta_local.contiguous(), group=process_group)
    theta_full = torch.cat(theta_list, dim=0)  # (total_K, D)

    # All-gather gradient norms
    G_norms_list = [torch.zeros_like(G_theta_norms_local) for _ in range(world_size)]
    dist.all_gather(G_norms_list, G_theta_norms_local.contiguous(), group=process_group)
    G_norms_full = torch.cat(G_norms_list)  # (total_K,)
    
    # Transpose for soft_ortho: expects (D, K) with vectors as columns
    theta_T = theta_full.T  # (D, total_K)
    
    # Compute logit bias from gradient norms
    logit_bias = torch.log(G_norms_full + 1e-8)
    
    # Apply soft orthogonalization
    theta_ortho_T = soft_ortho(
        theta_T, 
        num_iterations=num_iterations,
        temperature=temperature,
        logit_bias=logit_bias
    )  # (D, total_K)
    
    # Transpose back and slice local portion
    theta_ortho_full = theta_ortho_T.T  # (total_K, D)
    start_idx = rank * local_K
    end_idx = (rank + 1) * local_K
    
    return theta_ortho_full[start_idx:end_idx].contiguous()


# =============================================================================
# Main Distributed LoRA DCT Class
# =============================================================================

class DistributedLoRADCT:
    """
    Distributed LoRA DCT training.
    
    Each GPU owns (num_factors // world_size) complete LoRA factors.
    Training alternates between:
    1. Updating output directions U given current LoRA params
    2. Updating LoRA params given U via gradient ascent
    3. Soft orthogonalizing LoRA params across all GPUs
    4. Normalizing LoRA column norms
    """
    
    def __init__(
        self,
        num_factors: int,
        rank: int,
        world_size: int,
        config: LoRAFactorConfig,
        process_group=None,
    ):
        assert num_factors % world_size == 0, \
            f"num_factors ({num_factors}) must be divisible by world_size ({world_size})"

        self.num_factors = num_factors
        self.local_num_factors = num_factors // world_size
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.process_group = process_group  # None = default global group
        
        # Will be initialized in fit()
        self.lora_factors: Optional[LoRAFactorSet] = None
        self.sliced_model: Optional[BatchedSlicedLoRAModel] = None
        self.U_local: Optional[Tensor] = None
        
        # Training state
        self.objective_values = []
        self.device = None
        self.dtype = None
    
    def fit(
        self,
        model: nn.Module,
        X: List[Tensor],
        Y: List[Tensor],
        factor_batch_size: int = 16,
        init: str = "rand_backward",
        norm_value: float = 1.0,
        max_iters: int = 10,
        beta: float = 1.0,
        deflation: bool = True,
        soft_ortho_temp: float = 1.0,
        soft_ortho_iterations: int = 10,
        separate_u: bool = False,
        callback: Optional[Callable[[int, float], None]] = None,
        offload_factors: bool = True,
        compile: bool = False
    ) -> Tuple[LoRAFactorSet, Tensor]:
        """
        Train LoRA factors to maximize activation change at target layer.

        Args:
            model: The base model
            X: List of (S_i, D) tensors - source activations per sample
            Y: List of (S_i, D) tensors - target activations per sample
            factor_batch_size: Chunk size for processing factors
            init: Initialization strategy ("random" or "rand_backward")
            norm_value: Target norm for LoRA column normalization
            max_iters: Number of training iterations
            beta: Momentum parameter (1.0 = pure gradient, 0.0 = pure momentum)
            deflation: Whether to apply soft orthogonalization
            soft_ortho_temp: Temperature for soft orthogonalization
            soft_ortho_iterations: Number of soft ortho iterations
            separate_u: If True, let U vary per sample (||delta||_2 objective)
            callback: Optional callback function called after each iteration
                      with signature callback(iteration: int, objective_value: float)
            offload_factors: If True, store factors on CPU and load to GPU on-demand

        Returns:
            Tuple of (lora_factors, U_local)
        """
        assert init in ["random", "rand_backward"]

        N = len(X)
        D_source = X[0].shape[-1]
        D_target = Y[0].shape[-1]

        # Get device from model (X/Y may be on CPU for memory efficiency)
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.factor_batch_size = factor_batch_size
        self.norm_value = norm_value
        
        # Create LoRA factors
        self.lora_factors = LoRAFactorSet(
            self.local_num_factors, self.config, model, self.device, self.dtype,
            offload_to_cpu=offload_factors
        )
        
        # Create batched sliced model
        self.sliced_model = BatchedSlicedLoRAModel(model, self.config, self.lora_factors)

        if compile:
            if self.rank == 0:
                print("Compiling forward pass with torch.compile...")
            self.sliced_model.forward_chunk_delta_mean = torch.compile(
                self.sliced_model.forward_chunk_delta_mean,
                dynamic=True,
            )

        # Compute unsteered output Y if not provided (or verify it matches)
        # Y should be the unsteered output at target_layer
        
        # Initialize U (output directions): (D_target, local_K)
        if self.rank == 0:
            print(f"Initializing U randomly with shape ({D_target}, {self.local_num_factors})")
        
        # Generate on rank 0 and broadcast for determinism
        if self.rank == 0:
            U_full = F.normalize(
                torch.randn(D_target, self.num_factors, device=self.device, dtype=self.dtype),
                dim=0
            )
        else:
            U_full = torch.empty(D_target, self.num_factors, device=self.device, dtype=self.dtype)
        
        dist.broadcast(U_full, src=0, group=self.process_group)

        # Slice local portion
        start_idx = self.rank * self.local_num_factors
        end_idx = (self.rank + 1) * self.local_num_factors
        self.U_local = U_full[:, start_idx:end_idx].contiguous()
        
        del U_full
        
        # Initialize LoRA params
        if init == "rand_backward":
            self._init_rand_backward(X, Y)
        else:
            self._init_random()
        
        if self.rank == 0:
            print(f"Starting LoRA DCT training:")
            print(f"  Local factors: {self.local_num_factors}")
            print(f"  Total factors: {self.num_factors}")
            print(f"  LoRA rank: {self.config.lora_rank}")
            print(f"  Source layers: {self.config.source_layers}")
            print(f"  Target layer: {self.config.target_layer}")
            print(f"  Target modules: {self.config.target_modules}")
            print(f"  Flattened dim per factor: {self.lora_factors.flat_dim}")
        
        # Training loop
        for iteration in tqdm(range(max_iters), disable=(self.rank != 0)):
            obj_value = self._training_step(
                X, Y, beta, deflation,
                soft_ortho_temp, soft_ortho_iterations, separate_u
            )
            self.objective_values.append(obj_value)
            if callback is not None:
                callback(iteration + 1, obj_value)

        # Gather scores (gradient norms of final iteration)
        self._compute_final_scores(X, Y, separate_u)
        
        return self.lora_factors, self.U_local
    
    def _init_random(self):
        """Random initialization."""
        if self.rank == 0:
            print("Initializing LoRA params randomly...")
        
        # Generate full init on rank 0 and broadcast
        flat_dim = self.lora_factors.flat_dim
        
        if self.rank == 0:
            init_full = torch.randn(
                self.num_factors, flat_dim, 
                device=self.device, dtype=self.dtype
            ) * 0.01
        else:
            init_full = torch.empty(
                self.num_factors, flat_dim,
                device=self.device, dtype=self.dtype
            )
        
        dist.broadcast(init_full, src=0, group=self.process_group)

        # Slice local portion
        start_idx = self.rank * self.local_num_factors
        end_idx = (self.rank + 1) * self.local_num_factors
        init_local = init_full[start_idx:end_idx]
        
        self.lora_factors.set_from_flattened(init_local)
        self.lora_factors.normalize_columns(self.norm_value)
        
        del init_full
    
    def _init_rand_backward(self, X: List[Tensor], Y: List[Tensor]):
        """
        Initialize LoRA params using gradient-based approach.

        Start with small random init, compute gradients of <U, delta>,
        and set LoRA params proportional to normalized gradients.

        Args:
            X: List of (S_i, D) tensors - source activations per sample
            Y: List of (S_i, D) tensors - target activations per sample
        """
        if self.rank == 0:
            print("Initializing LoRA params via backward pass...")

        # Start with small random init
        self._init_random()

        # Run one forward-backward to get gradients
        G_lora_avg = StreamingAverage()

        K = self.local_num_factors
        N = len(X)
        for i in range(N):
            # Process each sample individually
            x_sample = X[i].unsqueeze(0).to(self.device)  # (1, S_i, D)
            y_sample = Y[i].unsqueeze(0).to(self.device)  # (1, S_i, D)

            self.lora_factors.zero_grad()

            # Process in chunks to avoid OOM - gradients accumulate across chunks
            for k_start in range(0, K, self.factor_batch_size):
                k_end = min(k_start + self.factor_batch_size, K)

                # Forward and compute delta_mean for this chunk: (k_chunk, D)
                delta_mean_chunk = self.sliced_model.forward_chunk_delta_mean(
                    x_sample, y_sample, k_start, k_end,
                    self.config.target_position_indices
                )

                # Objective: <delta, U> for this chunk
                # delta_mean_chunk: (k_chunk, D), U_local[:, k_start:k_end]: (D, k_chunk)
                # Move U slice to delta's device (preserves autograd through delta)
                U_slice = self.U_local[:, k_start:k_end].T.to(delta_mean_chunk.device)
                obj_chunk = (delta_mean_chunk * U_slice).sum()
                obj_chunk.backward()  # gradients accumulate

            G_lora_batch = self.lora_factors.get_grad_flattened()  # (K, D_lora)
            G_lora_avg.update(G_lora_batch.unsqueeze(0))

        # Set LoRA params to normalized gradients
        G_lora = G_lora_avg.get_mean()  # (1, K, D_lora) -> need to squeeze first dim
        if G_lora.dim() == 3:
            G_lora = G_lora.squeeze(0)  # (K, D_lora)
        # Ensure 2D even if K=1
        if G_lora.dim() == 1:
            G_lora = G_lora.unsqueeze(0)  # (1, D_lora)
        G_lora_normalized = F.normalize(G_lora, dim=1)

        self.lora_factors.set_from_flattened(G_lora_normalized)
        self.lora_factors.normalize_columns(self.norm_value)
    
    def _training_step(
        self,
        X: List[Tensor],
        Y: List[Tensor],
        beta: float,
        deflation: bool,
        soft_ortho_temp: float,
        soft_ortho_iterations: int,
        separate_u: bool
    ) -> float:
        """
        Single training iteration.

        Args:
            X: List of (S_i, D) tensors - source activations per sample
            Y: List of (S_i, D) tensors - target activations per sample
            beta: Momentum parameter
            deflation: Whether to apply soft orthogonalization
            soft_ortho_temp: Temperature for soft orthogonalization
            soft_ortho_iterations: Number of soft ortho iterations
            separate_u: If True, let U vary per sample

        Returns:
            Average objective value across samples
        """
        K = self.local_num_factors
        D = self.U_local.shape[0]
        N = len(X)

        # Accumulators
        G_U_avg = StreamingAverage()
        G_lora_avg = StreamingAverage()
        obj_avg = StreamingAverage()

        # Process each sample individually
        for i in range(N):
            x_sample = X[i].unsqueeze(0).to(self.device)  # (1, S_i, D)
            y_sample = Y[i].unsqueeze(0).to(self.device)  # (1, S_i, D)

            self.lora_factors.zero_grad()

            # Accumulators for this sample (to build full K-sized tensors from chunks)
            G_U_chunks = []
            obj_sum = 0.0

            # Process in chunks to avoid OOM - gradients accumulate across chunks
            for k_start in range(0, K, self.factor_batch_size):
                k_end = min(k_start + self.factor_batch_size, K)

                # Forward and compute delta_mean for this chunk: (k_chunk, D)
                delta_mean_chunk = self.sliced_model.forward_chunk_delta_mean(
                    x_sample, y_sample, k_start, k_end,
                    self.config.target_position_indices
                )

                if separate_u:
                    # Objective: ||delta||_2 for each factor
                    delta_norms_chunk = delta_mean_chunk.norm(dim=1)  # (k_chunk,)
                    obj_chunk = delta_norms_chunk.sum()
                    obj_chunk.backward()  # gradients accumulate

                    # G_U is the normalized delta (direction of max change)
                    G_U_chunk = delta_mean_chunk.detach() / (delta_norms_chunk.unsqueeze(1) + 1e-8)  # (k_chunk, D)
                    obj_sum += delta_norms_chunk.sum().item()
                else:
                    # Objective: <delta, U> for each factor
                    U_slice = self.U_local[:, k_start:k_end].T.to(delta_mean_chunk.device)
                    dot_products_chunk = (delta_mean_chunk * U_slice).sum(dim=1)  # (k_chunk,)
                    obj_chunk = dot_products_chunk.sum()
                    obj_chunk.backward()  # gradients accumulate

                    G_U_chunk = delta_mean_chunk.detach()  # (k_chunk, D)
                    obj_sum += dot_products_chunk.sum().item()

                G_U_chunks.append(G_U_chunk)

            # Concatenate G_U chunks: list of (k_chunk, D) -> (K, D)
            G_U_batch = torch.cat(G_U_chunks, dim=0)
            obj_value = obj_sum / K

            G_lora_batch = self.lora_factors.get_grad_flattened()  # (K, D_lora)

            # Update accumulators (move to self.device for pipeline-parallel models)
            G_U_avg.update(G_U_batch.to(self.device).T.unsqueeze(0))  # (1, D, K)
            G_lora_avg.update(G_lora_batch.unsqueeze(0))  # (1, K, D_lora)
            obj_avg.update(torch.tensor([[obj_value]], device=self.device))

        # === Update U ===
        G_U = G_U_avg.get_mean()
        if G_U.dim() == 3:
            G_U = G_U.squeeze(0)  # (D, K)
        # Ensure 2D even if K=1
        if G_U.dim() == 1:
            G_U = G_U.unsqueeze(1)  # (D, 1)
        self.U_local = F.normalize(
            beta * G_U + (1 - beta) * self.U_local,
            dim=0
        )

        # === Update LoRA params ===
        G_lora = G_lora_avg.get_mean()
        if G_lora.dim() == 3:
            G_lora = G_lora.squeeze(0)  # (K, D_lora)
        # Ensure 2D even if K=1
        if G_lora.dim() == 1:
            G_lora = G_lora.unsqueeze(0)  # (1, D_lora)
        current_lora = self.lora_factors.get_flattened_all()  # (K, D_lora)

        new_lora = F.normalize(
            beta * G_lora + (1 - beta) * current_lora,
            dim=1
        )

        # === Distributed soft orthogonalization ===
        if deflation:
            G_lora_norms = G_lora.norm(dim=1)  # (K,)
            new_lora = distributed_soft_ortho_lora(
                new_lora, G_lora_norms,
                self.rank, self.world_size,
                num_iterations=soft_ortho_iterations,
                temperature=soft_ortho_temp,
                process_group=self.process_group,
            )

        self.lora_factors.set_from_flattened(new_lora)

        # === Normalize columns ===
        self.lora_factors.normalize_columns(self.norm_value)

        return obj_avg.get_mean().item()
    
    def _compute_final_scores(
        self,
        X: List[Tensor],
        Y: List[Tensor],
        separate_u: bool
    ):
        """
        Compute final scores for ranking factors.

        Args:
            X: List of (S_i, D) tensors - source activations per sample
            Y: List of (S_i, D) tensors - target activations per sample
            separate_u: If True, use ||delta||_2 objective
        """
        K = self.local_num_factors
        N = len(X)

        G_lora_avg = StreamingAverage()

        # Process each sample individually
        for i in range(N):
            x_sample = X[i].unsqueeze(0).to(self.device)  # (1, S_i, D)
            y_sample = Y[i].unsqueeze(0).to(self.device)  # (1, S_i, D)

            self.lora_factors.zero_grad()

            with torch.no_grad():
                # Process in chunks to avoid OOM
                scores_chunks = []
                for k_start in range(0, K, self.factor_batch_size):
                    k_end = min(k_start + self.factor_batch_size, K)

                    # Forward and compute delta_mean for this chunk: (k_chunk, D)
                    delta_mean_chunk = self.sliced_model.forward_chunk_delta_mean(
                        x_sample, y_sample, k_start, k_end,
                        self.config.target_position_indices
                    )

                    if separate_u:
                        scores_chunk = delta_mean_chunk.norm(dim=1)  # (k_chunk,)
                    else:
                        U_slice = self.U_local[:, k_start:k_end].T.to(delta_mean_chunk.device)
                        scores_chunk = (delta_mean_chunk * U_slice).sum(dim=1)  # (k_chunk,)

                    scores_chunks.append(scores_chunk)

                # Concatenate scores: (K,)
                scores_batch = torch.cat(scores_chunks, dim=0)

            G_lora_avg.update(scores_batch.unsqueeze(0))

        # Local scores
        local_scores = G_lora_avg.get_mean()
        if local_scores.dim() == 2:
            local_scores = local_scores.squeeze(0)  # (local_K,)
        # Ensure at least 1D for all_gather
        if local_scores.dim() == 0:
            local_scores = local_scores.unsqueeze(0)  # (1,)

        # All-gather to get global scores
        scores_list = [torch.zeros_like(local_scores) for _ in range(self.world_size)]
        dist.all_gather(scores_list, local_scores.contiguous(), group=self.process_group)
        self.scores = torch.cat(scores_list)  # (total_K,)

    def save(self, output_dir: str, prefix: str = "lora_dct"):
        """Save trained factors and metadata."""
        pg = self.process_group
        if self.rank == 0:
            os.makedirs(output_dir, exist_ok=True)

        dist.barrier(group=pg)

        # Gather all factors on rank 0
        local_flat = self.lora_factors.get_flattened_all()  # (local_K, D)

        if self.rank == 0:
            flat_list = [torch.zeros_like(local_flat) for _ in range(self.world_size)]
            dist.gather(local_flat.contiguous(), flat_list, dst=0, group=pg)
            all_flat = torch.cat(flat_list, dim=0)
        else:
            dist.gather(local_flat.contiguous(), dst=0, group=pg)

        # Gather U
        if self.rank == 0:
            U_list = [torch.zeros_like(self.U_local) for _ in range(self.world_size)]
            dist.gather(self.U_local.contiguous(), U_list, dst=0, group=pg)
            all_U = torch.cat(U_list, dim=1)
        else:
            dist.gather(self.U_local.contiguous(), dst=0, group=pg)

        if self.rank == 0:
            # Save flattened parameters
            torch.save(all_flat, os.path.join(output_dir, f"{prefix}_all_factors.pt"))
            torch.save(all_U, os.path.join(output_dir, f"{prefix}_U.pt"))
            torch.save(self.scores, os.path.join(output_dir, f"{prefix}_scores.pt"))
            torch.save(
                torch.tensor(self.objective_values),
                os.path.join(output_dir, f"{prefix}_objective_values.pt")
            )
            
            # Save individual PEFT-compatible state dicts for top factors
            # Create a temporary full LoRA factor set for export
            # For now, save the layout info for reconstruction
            import json
            layout_info = {
                'num_factors': self.num_factors,
                'local_num_factors': self.local_num_factors,
                'lora_rank': self.config.lora_rank,
                'source_layers': list(self.config.source_layers),
                'target_layer': self.config.target_layer,
                'target_modules': self.config.target_modules,
                'norm_value': self.norm_value,
                'param_layout': self.lora_factors._param_layout,
                'world_size': self.world_size,
            }
            with open(os.path.join(output_dir, f"{prefix}_config.json"), 'w') as f:
                json.dump(layout_info, f, indent=2, default=str)
            
            print(f"Saved {self.num_factors} LoRA factors to {output_dir}")

        dist.barrier(group=pg)

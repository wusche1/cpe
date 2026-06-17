"""
LoRA DCT: Low-Rank Adapter Deep Causal Transcoding

A distributed implementation for learning LoRA adapters that maximize
activation changes at target layers using softly orthogonalized gradient iteration.
"""

from .lora_dct_distributed import (
    # Configuration
    LoRAFactorConfig,
    
    # Core classes
    LoRAFactorSet,
    BatchedSlicedLoRAModel,
    DistributedLoRADCT,
    
    # Utility functions
    soft_ortho,
    distributed_soft_ortho_lora,
    apply_batched_lora,
    StreamingAverage,
)

from .peft_export import (
    load_lora_dct_results,
    reconstruct_lora_params,
    export_factor_to_peft,
    export_top_k_factors,
    create_peft_config,
    load_factor_into_model,
    combine_factors,
)

__version__ = "0.1.0"
__all__ = [
    # Config
    "LoRAFactorConfig",
    
    # Core
    "LoRAFactorSet",
    "BatchedSlicedLoRAModel",
    "DistributedLoRADCT",
    
    # Utilities
    "soft_ortho",
    "distributed_soft_ortho_lora",
    "apply_batched_lora",
    "StreamingAverage",
    
    # PEFT Export
    "load_lora_dct_results",
    "reconstruct_lora_params",
    "export_factor_to_peft",
    "export_top_k_factors",
    "create_peft_config",
    "load_factor_into_model",
    "combine_factors",
]

"""
PEFT Export Utilities for LoRA DCT

Utilities for exporting trained LoRA factors to PEFT-compatible format
and loading them for inference.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn as nn

try:
    import safetensors.torch
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


@dataclass
class LoRAExportConfig:
    """Configuration for exporting LoRA factors."""
    source_layers: Tuple[int, int]
    target_modules: List[str]
    lora_rank: int
    lora_alpha: float = 1.0
    lora_dropout: float = 0.0


def load_lora_dct_results(output_dir: str, prefix: str = "lora_dct") -> Dict[str, Any]:
    """
    Load trained LoRA DCT results.
    
    Args:
        output_dir: Directory containing saved results
        prefix: Prefix used when saving
        
    Returns:
        Dict containing:
            - 'all_factors': (K, D) flattened LoRA params
            - 'U': (D_target, K) output directions
            - 'scores': (K,) factor scores
            - 'config': configuration dict
            - 'objective_values': training objective history
    """
    results = {}
    
    # Load tensors
    results['all_factors'] = torch.load(
        os.path.join(output_dir, f"{prefix}_all_factors.pt"),
        map_location='cpu',
        weights_only=True
    )
    results['U'] = torch.load(
        os.path.join(output_dir, f"{prefix}_U.pt"),
        map_location='cpu',
        weights_only=True
    )
    results['scores'] = torch.load(
        os.path.join(output_dir, f"{prefix}_scores.pt"),
        map_location='cpu',
        weights_only=True
    )
    results['objective_values'] = torch.load(
        os.path.join(output_dir, f"{prefix}_objective_values.pt"),
        map_location='cpu',
        weights_only=True
    )
    
    # Load config
    with open(os.path.join(output_dir, f"{prefix}_config.json"), 'r') as f:
        results['config'] = json.load(f)
    
    return results


def reconstruct_lora_params(
    flattened: Tensor,
    param_layout: List[Dict],
    num_factors: int
) -> Dict[str, Dict[str, Tensor]]:
    """
    Reconstruct A and B matrices from flattened representation.
    
    Args:
        flattened: (K, D) flattened parameters
        param_layout: Layout information from config
        num_factors: Number of factors
        
    Returns:
        Dict mapping key -> {'A': tensor, 'B': tensor}
    """
    params = {}
    offset = 0
    
    for layout in param_layout:
        key = layout['key']
        A_shape = tuple(layout['A_shape'])
        B_shape = tuple(layout['B_shape'])
        A_numel = layout['A_numel']
        B_numel = layout['B_numel']
        
        A = flattened[:, offset:offset+A_numel].reshape(num_factors, *A_shape[1:])
        offset += A_numel
        
        B = flattened[:, offset:offset+B_numel].reshape(num_factors, *B_shape[1:])
        offset += B_numel
        
        params[key] = {'A': A, 'B': B}
    
    return params


def export_factor_to_peft(
    factor_idx: int,
    flattened: Tensor,
    config: Dict[str, Any],
    adapter_name: str = "default",
    for_vllm: bool = False
) -> Dict[str, Tensor]:
    """
    Export a single factor to PEFT-compatible state dict.

    Args:
        factor_idx: Index of the factor to export
        flattened: (K, D) flattened parameters for all factors
        config: Configuration dict from training
        adapter_name: PEFT adapter name (ignored if for_vllm=True)
        for_vllm: If True, use vLLM-compatible key format (no adapter name)

    Returns:
        State dict compatible with PEFT's load_state_dict or vLLM
    """
    param_layout = config['param_layout']
    source_layers = config['source_layers']
    target_modules = config['target_modules']

    state = {}
    offset = 0

    for layout in param_layout:
        key = layout['key']
        A_shape = tuple(layout['A_shape'])
        B_shape = tuple(layout['B_shape'])
        A_numel = layout['A_numel']
        B_numel = layout['B_numel']

        # Extract this factor's A and B
        A = flattened[factor_idx, offset:offset+A_numel].reshape(*A_shape[1:])
        offset += A_numel

        B = flattened[factor_idx, offset:offset+B_numel].reshape(*B_shape[1:])
        offset += B_numel

        # Parse key to get layer_idx and module_name
        # key format: "layer{idx}_{module_name}"
        parts = key.split('_', 1)
        layer_idx = int(parts[0].replace('layer', ''))
        module_name = parts[1]

        # PEFT naming convention
        prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{module_name}"

        if for_vllm:
            # vLLM expects simple format without adapter name
            state[f"{prefix}.lora_A.weight"] = A.clone()
            state[f"{prefix}.lora_B.weight"] = B.clone()
        else:
            # Standard PEFT format with adapter name
            state[f"{prefix}.lora_A.{adapter_name}.weight"] = A.clone()
            state[f"{prefix}.lora_B.{adapter_name}.weight"] = B.clone()

    return state


def export_top_k_factors(
    output_dir: str,
    prefix: str = "lora_dct",
    k: int = 10,
    export_dir: Optional[str] = None,
    adapter_name: str = "default"
) -> List[str]:
    """
    Export top-k factors (by score) to individual PEFT state dicts.
    
    Args:
        output_dir: Directory containing trained results
        prefix: Prefix used during training
        k: Number of top factors to export
        export_dir: Directory to save exports (defaults to output_dir/peft_exports)
        adapter_name: PEFT adapter name
        
    Returns:
        List of paths to exported state dicts
    """
    # Load results
    results = load_lora_dct_results(output_dir, prefix)
    
    flattened = results['all_factors']
    scores = results['scores']
    config = results['config']
    
    # Get top-k by score
    top_k_indices = torch.argsort(scores, descending=True)[:k]
    
    # Setup export directory
    if export_dir is None:
        export_dir = os.path.join(output_dir, "peft_exports")
    os.makedirs(export_dir, exist_ok=True)
    
    # Export each factor
    export_paths = []
    for rank_idx, factor_idx in enumerate(top_k_indices):
        factor_idx = factor_idx.item()
        score = scores[factor_idx].item()
        
        state = export_factor_to_peft(factor_idx, flattened, config, adapter_name)
        
        path = os.path.join(export_dir, f"factor_{factor_idx}_rank_{rank_idx}_score_{score:.4f}.pt")
        torch.save(state, path)
        export_paths.append(path)
        
        print(f"Exported factor {factor_idx} (rank {rank_idx}, score {score:.4f}) to {path}")
    
    # Save export metadata
    export_meta = {
        'source_config': config,
        'exported_factors': [
            {
                'factor_idx': top_k_indices[i].item(),
                'rank': i,
                'score': scores[top_k_indices[i]].item(),
                'path': export_paths[i]
            }
            for i in range(len(top_k_indices))
        ],
        'adapter_name': adapter_name,
    }
    
    with open(os.path.join(export_dir, "export_metadata.json"), 'w') as f:
        json.dump(export_meta, f, indent=2)
    
    return export_paths


def create_peft_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a PEFT LoraConfig-compatible dict from LoRA DCT config.

    Args:
        config: LoRA DCT config dict

    Returns:
        Dict that can be used with LoraConfig(**config)
    """
    return {
        'r': config['lora_rank'],
        'lora_alpha': config.get('lora_alpha', config['lora_rank']),
        'target_modules': config['target_modules'],
        'lora_dropout': 0.0,
        'bias': 'none',
        'task_type': 'CAUSAL_LM',
        'layers_to_transform': list(range(
            config['source_layers'][0],
            config['source_layers'][1] + 1
        )),
    }


def create_peft_adapter_config(config: Dict[str, Any], base_model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a PEFT adapter_config.json compatible dict for vLLM.

    Args:
        config: LoRA DCT config dict
        base_model_name: Optional base model name

    Returns:
        Dict that can be saved as adapter_config.json
    """
    return {
        'alpha_pattern': {},
        'auto_mapping': None,
        'base_model_name_or_path': base_model_name or '',
        'bias': 'none',
        'fan_in_fan_out': False,
        'inference_mode': True,
        'init_lora_weights': True,
        'layer_replication': None,
        'layers_pattern': None,
        'layers_to_transform': list(range(
            config['source_layers'][0],
            config['source_layers'][1] + 1
        )),
        'loftq_config': {},
        'lora_alpha': config.get('lora_alpha', config['lora_rank']),
        'lora_dropout': 0.0,
        'megatron_config': None,
        'megatron_core': 'megatron.core',
        'modules_to_save': None,
        'peft_type': 'LORA',
        'r': config['lora_rank'],
        'rank_pattern': {},
        'revision': None,
        'target_modules': config['target_modules'],
        'task_type': 'CAUSAL_LM',
        'use_dora': False,
        'use_rslora': False,
    }


def export_factor_to_peft_dir(
    factor_idx: int,
    results: Dict[str, Any],
    output_dir: str,
    base_model_name: Optional[str] = None,
    for_vllm: bool = True
) -> str:
    """
    Export a single factor to PEFT directory format for vLLM.

    This creates a directory with:
    - adapter_config.json: PEFT configuration
    - adapter_model.safetensors (or .bin): LoRA weights

    Args:
        factor_idx: Index of the factor to export
        results: Results dict from load_lora_dct_results()
        output_dir: Base directory to create factor subdirectory in
        base_model_name: Optional base model name for config
        for_vllm: If True (default), use vLLM-compatible key format

    Returns:
        Path to the created factor directory
    """
    flattened = results['all_factors']
    config = results['config']

    # Create factor directory
    factor_dir = os.path.join(output_dir, f"factor_{factor_idx}")
    os.makedirs(factor_dir, exist_ok=True)

    # Get state dict for this factor (use vLLM format by default)
    state_dict = export_factor_to_peft(factor_idx, flattened, config, for_vllm=for_vllm)

    # Convert state dict for compatibility
    standalone_state = {}
    for key, value in state_dict.items():
        # Convert to float16 for vLLM compatibility
        standalone_state[key] = value.half().contiguous()

    # Save weights
    if HAS_SAFETENSORS:
        safetensors.torch.save_file(
            standalone_state,
            os.path.join(factor_dir, "adapter_model.safetensors")
        )
    else:
        torch.save(standalone_state, os.path.join(factor_dir, "adapter_model.bin"))

    # Save adapter config
    adapter_config = create_peft_adapter_config(config, base_model_name)
    with open(os.path.join(factor_dir, "adapter_config.json"), 'w') as f:
        json.dump(adapter_config, f, indent=2)

    return factor_dir


def _flatten_spectrum(A: Tensor, B: Tensor) -> Tuple[Tensor, Tensor]:
    """Replace the singular values of B @ A with ones (flatten the spectrum).

    Works in r×r space to avoid materialising the full (d_out, d_in) product.
    """
    Q_B, R_B = torch.linalg.qr(B)
    Q_A, R_A = torch.linalg.qr(A.T)
    M = R_B @ R_A.T
    U, S, Vt = torch.linalg.svd(M)
    return Vt @ Q_A.T, Q_B @ U          # A_new, B_new


def export_concatenated_factors_to_peft_dir(
    factor_indices: List[int],
    results: Dict[str, Any],
    output_dir: str,
    base_model_name: Optional[str] = None,
    composition_mode: str = "normalize",
) -> str:
    """
    Export multiple factors concatenated into a single higher-rank PEFT adapter.

    For each weight key, concatenates:
    - lora_A.weight: along dim 0 (r, d_in) -> (i*r, d_in)
    - lora_B.weight: along dim 1 (d_out, r) -> (d_out, i*r)

    Args:
        factor_indices: Indices of factors to concatenate.
        results: Results dict from load_lora_dct_results().
        output_dir: Directory to write the adapter files to.
        base_model_name: Base model name for adapter_config.json.
        composition_mode: How to combine factors after concatenation.
            "normalize" (default): divide A by n so the adapter is the average effect.
            "flatten": replace singular values of each B@A with ones.
            "sum": raw concatenation (no post-processing).

    Returns:
        Path to the created adapter directory.
    """
    flattened = results['all_factors']
    config = results['config']

    # Get individual state dicts for each factor (vLLM format)
    state_dicts = [
        export_factor_to_peft(idx, flattened, config, for_vllm=True)
        for idx in factor_indices
    ]

    # Concatenate A and B matrices across factors
    concatenated = {}
    for key in state_dicts[0].keys():
        tensors = [sd[key] for sd in state_dicts]
        if '.lora_A.weight' in key:
            # A is (r, d_in), concatenate along dim 0 -> (i*r, d_in)
            concatenated[key] = torch.cat(tensors, dim=0)
        elif '.lora_B.weight' in key:
            # B is (d_out, r), concatenate along dim 1 -> (d_out, i*r)
            concatenated[key] = torch.cat(tensors, dim=1)
        else:
            concatenated[key] = tensors[0]

    # Apply composition mode
    n = len(factor_indices)
    if composition_mode == "normalize" and n > 1:
        for key in concatenated:
            if '.lora_A.weight' in key:
                concatenated[key] = concatenated[key] / n
    elif composition_mode == "flatten":
        a_keys = sorted(k for k in concatenated if '.lora_A.weight' in k)
        b_keys = sorted(k for k in concatenated if '.lora_B.weight' in k)
        for a_key, b_key in zip(a_keys, b_keys):
            A_new, B_new = _flatten_spectrum(concatenated[a_key], concatenated[b_key])
            concatenated[a_key] = A_new
            concatenated[b_key] = B_new

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert to float16 for vLLM compatibility
    standalone_state = {k: v.half().contiguous() for k, v in concatenated.items()}

    # Save weights
    if HAS_SAFETENSORS:
        safetensors.torch.save_file(
            standalone_state,
            os.path.join(output_dir, "adapter_model.safetensors")
        )
    else:
        torch.save(standalone_state, os.path.join(output_dir, "adapter_model.bin"))

    # Save adapter config with updated rank
    concatenated_rank = config['lora_rank'] * len(factor_indices)
    adapter_config = create_peft_adapter_config(config, base_model_name)
    adapter_config['r'] = concatenated_rank
    adapter_config['lora_alpha'] = concatenated_rank

    with open(os.path.join(output_dir, "adapter_config.json"), 'w') as f:
        json.dump(adapter_config, f, indent=2)

    return output_dir


# TODO: adapter_name is passed as the for_vllm bool arg to export_factor_to_peft_dir below.
# The adapter_name parameter is effectively ignored. Fix the call or remove the parameter.
def export_all_factors_to_peft_dirs(
    results: Dict[str, Any],
    output_dir: str,
    factor_indices: Optional[List[int]] = None,
    base_model_name: Optional[str] = None,
    adapter_name: str = "default"
) -> Dict[int, str]:
    """
    Export multiple factors to PEFT directory format.

    Args:
        results: Results dict from load_lora_dct_results()
        output_dir: Base directory to create factor subdirectories in
        factor_indices: Indices of factors to export (None = all)
        base_model_name: Optional base model name for config
        adapter_name: PEFT adapter name

    Returns:
        Dict mapping factor_idx to directory path
    """
    num_factors = results['all_factors'].shape[0]

    if factor_indices is None:
        factor_indices = list(range(num_factors))

    paths = {}
    for idx in factor_indices:
        paths[idx] = export_factor_to_peft_dir(
            idx, results, output_dir, base_model_name, adapter_name
        )

    return paths


def load_factor_into_model(
    model: nn.Module,
    state_dict_path: str,
    strict: bool = False
):
    """
    Load a PEFT-exported factor into a model with existing LoRA layers.
    
    This assumes the model already has PEFT LoRA layers added via get_peft_model.
    
    Args:
        model: PeftModel with LoRA layers
        state_dict_path: Path to exported state dict
        strict: Whether to require all keys to match
    """
    state = torch.load(state_dict_path, map_location='cpu')
    
    # Load with strict=False by default since we're only loading a subset
    missing, unexpected = model.load_state_dict(state, strict=strict)
    
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
    
    return model


def combine_factors(
    output_dir: str,
    factor_indices: List[int],
    weights: Optional[List[float]] = None,
    prefix: str = "lora_dct",
    adapter_name: str = "default"
) -> Dict[str, Tensor]:
    """
    Combine multiple factors into a single LoRA adapter.
    
    Args:
        output_dir: Directory containing trained results
        factor_indices: Indices of factors to combine
        weights: Optional weights for each factor (defaults to uniform)
        prefix: Prefix used during training
        adapter_name: PEFT adapter name
        
    Returns:
        Combined state dict
    """
    results = load_lora_dct_results(output_dir, prefix)
    flattened = results['all_factors']
    config = results['config']
    
    if weights is None:
        weights = [1.0 / len(factor_indices)] * len(factor_indices)
    
    assert len(weights) == len(factor_indices)
    
    # Get state dicts for each factor
    states = [
        export_factor_to_peft(idx, flattened, config, adapter_name)
        for idx in factor_indices
    ]
    
    # Combine by weighted average
    combined = {}
    for key in states[0].keys():
        combined[key] = sum(w * s[key] for w, s in zip(weights, states))
    
    return combined


# Example usage functions
def example_apply_lora_to_model():
    """Example of how to apply exported LoRA factors to a model."""
    example_code = '''
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from peft_export import load_lora_dct_results, export_factor_to_peft, create_peft_config
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
    
    # Load LoRA DCT results
    results = load_lora_dct_results("./outputs", "lora_dct")
    
    # Create PEFT config
    peft_config_dict = create_peft_config(results['config'])
    peft_config = LoraConfig(**peft_config_dict)
    
    # Add LoRA layers to model
    model = get_peft_model(model, peft_config)
    
    # Load a specific factor (e.g., the top-scoring one)
    top_factor_idx = results['scores'].argmax().item()
    state = export_factor_to_peft(top_factor_idx, results['all_factors'], results['config'])
    
    # Load into model
    model.load_state_dict(state, strict=False)
    
    # Now model has the LoRA factor applied
    # For inference:
    output = model.generate(input_ids, max_new_tokens=100)
    '''
    print(example_code)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PEFT Export Utilities")
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory containing trained results')
    parser.add_argument('--prefix', type=str, default='lora_dct',
                        help='Prefix used during training')
    parser.add_argument('--export_top_k', type=int, default=10,
                        help='Number of top factors to export')
    parser.add_argument('--export_dir', type=str, default=None,
                        help='Directory to save exports')
    
    args = parser.parse_args()
    
    export_top_k_factors(
        args.output_dir,
        args.prefix,
        args.export_top_k,
        args.export_dir
    )

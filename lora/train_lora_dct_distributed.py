#!/usr/bin/env python
# coding: utf-8
"""
Distributed LoRA DCT Training Script

Train LoRA factors that maximize activation changes using Deep Causal Transcoding.

Usage:
    torchrun --nproc_per_node=4 train_lora_dct_distributed.py \
        --model_name Qwen/Qwen3-MoE-15B-A2B \
        --dataset /path/to/dataset \
        --source_layer_start 8 \
        --source_layer_end 12 \
        --target_layer 20 \
        --num_factors 256 \
        --output_dir ./outputs
"""

import os
import sys
import json
import argparse
import math
import gc
import time
import datetime

import torch
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset, load_from_disk

# Optional wandb import
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Import our LoRA DCT implementation
from lora_dct_distributed import (
    LoRAFactorConfig,
    LoRAFactorSet,
    BatchedSlicedLoRAModel,
    DistributedLoRADCT,
)

def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(hours=12))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def trim_model_for_training(model, source_layer_start: int, target_layer: int, rank: int):
    """
    Delete unused model components after activation caching to free GPU memory.

    Only keeps layers [source_layer_start, target_layer] and rotary_emb.
    """
    layers = model.model.layers
    num_layers = len(layers)

    # Count what we're deleting
    deleted_before = source_layer_start
    deleted_after = num_layers - target_layer - 1
    kept = target_layer - source_layer_start + 1

    if rank == 0:
        print(f"Trimming model: keeping layers [{source_layer_start}, {target_layer}] "
              f"({kept} layers), deleting {deleted_before + deleted_after} layers")

    # Delete layers before source
    for i in range(source_layer_start):
        layers[i] = None

    # Delete layers after target
    for i in range(target_layer + 1, num_layers):
        layers[i] = None

    # Delete embedding layer
    if hasattr(model.model, 'embed_tokens'):
        model.model.embed_tokens = None

    # Delete final norm
    if hasattr(model.model, 'norm'):
        model.model.norm = None

    # Delete LM head
    if hasattr(model, 'lm_head'):
        model.lm_head = None

    # Force garbage collection and clear CUDA cache
    gc.collect()
    torch.cuda.empty_cache()

    if rank == 0:
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory after trimming: {allocated:.2f} GB")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train distributed LoRA DCT models')
    
    # Model configuration
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-MoE-15B-A2B",
                        help='Model name to use')
    parser.add_argument('--tokenizer_name', type=str, default=None,
                        help='Tokenizer name (defaults to model_name)')
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, required=True,
                        help='HuggingFace dataset path')
    parser.add_argument('--split', type=str, default=None,
                        help='Dataset split to use')
    parser.add_argument('--field', type=str, default="prompt",
                        help='Field from dataset to use as instructions')
    parser.add_argument('--num_samples', type=int, default=32,
                        help='Number of samples to use')
    parser.add_argument('--system_prompt', type=str,
                        default="You are a helpful and harmless assistant.",
                        help='System prompt')
    parser.add_argument('--max_length', type=int, default=None,
                        help='Max sequence length for truncation')
    
    # Layer configuration
    parser.add_argument('--source_layer_start', type=int, default=8,
                        help='Start of source layer range (inclusive)')
    parser.add_argument('--source_layer_end', type=int, default=12,
                        help='End of source layer range (inclusive)')
    parser.add_argument('--target_layer', type=int, default=20,
                        help='Target layer to measure activation changes')
    
    # LoRA configuration
    parser.add_argument('--target_modules', type=str, default="o_proj",
                        help='Comma-separated list of target modules (e.g., "q_proj,k_proj,v_proj,o_proj")')
    parser.add_argument('--lora_rank', type=int, default=1,
                        help='LoRA rank')
    parser.add_argument('--norm_value', type=float, default=1.0,
                        help='Target norm for LoRA column normalization')
    
    # Training configuration
    parser.add_argument('--num_factors', type=int, default=256,
                        help='Total number of LoRA factors to learn')
    parser.add_argument('--num_iters', type=int, default=30,
                        help='Number of training iterations')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Momentum parameter (1.0 = pure gradient)')
    parser.add_argument('--forward_batch_size', type=int, default=1,
                        help='Batch size for initial forward passes')
    parser.add_argument('--backward_batch_size', type=int, default=1,
                        help='Batch size for training')
    parser.add_argument('--factor_batch_size', type=int, default=16,
                        help='Chunk size for processing factors')
    parser.add_argument('--no_offload_factors', action='store_true',
                        help='Disable CPU offloading of factors (default: offload enabled)')
    
    # Deflation/orthogonalization
    parser.add_argument('--deflate', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable soft orthogonalization')
    parser.add_argument('--deflation_iterations', type=int, default=10,
                        help='Number of deflation iterations')
    parser.add_argument('--deflation_temp', type=float, default=1.0,
                        help='Deflation temperature')
    
    # Target positions
    parser.add_argument('--token_idxs', type=str, default="-3:",
                        help='Target token positions (Python slice notation)')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default="./outputs",
                        help='Directory to save outputs')
    parser.add_argument('--output_prefix', type=str, default="lora_dct",
                        help='Prefix for output files')
    
    # Training modes
    parser.add_argument('--separate_u', action=argparse.BooleanOptionalAction, default=False,
                        help='Let output directions vary per sample')
    parser.add_argument('--init', type=str, default="rand_backward",
                        choices=["random", "rand_backward"],
                        help='Initialization strategy')
    
    # Data selection
    parser.add_argument('--data_start', type=int, default=None,
                        help='Start index in dataset')
    parser.add_argument('--data_end', type=int, default=None,
                        help='End index in dataset')
    
    # Seed
    parser.add_argument('--seed', type=int, default=325,
                        help='Random seed')

    # Chat template config
    parser.add_argument('--enable_thinking', action='store_true',
                        help='Enable thinking/reasoning in chat template (for supported models)')
    parser.add_argument('--reasoning_effort', type=str, default=None,
                        choices=[None, 'low', 'medium', 'high'],
                        help="gpt-oss reasoning effort (low/medium/high). If unset, chat-template default applies (medium for gpt-oss).")

    # "tp_size" = GPUs the model is sharded across per factor-group. NOTE: this is a
    # BAD NAME — it is PIPELINE parallelism (a device_map layer split), NOT tensor
    # parallelism, and it shards model WEIGHTS (factors are the data-parallel dim,
    # split across world_size//tp_size groups). TODO: rename to e.g. --model_shard_gpus.
    parser.add_argument('--training_tp_size', type=int, default=1,
                        help='GPUs per model shard (>1 = device_map PIPELINE split for large '
                             'models; misnomer, not tensor-parallel). Factors split across '
                             'world_size//tp_size groups.')

    # Compilation
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile for training forward pass')

    # Logging
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='lora-dct',
                        help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name (auto-generated if not provided)')

    return parser.parse_args()


def parse_token_idxs(token_idxs_str: str):
    """Parse token indices from string to slice or int."""
    parts = token_idxs_str.split(':')
    if len(parts) == 1:
        return int(parts[0])
    else:
        start = None if parts[0] == '' else int(parts[0])
        end = None if parts[1] == '' else int(parts[1])
        return slice(start, end)


def save_objective_plot(objective_values, output_dir, prefix, args, rank, use_wandb=False):
    """Save plot of objective values during training."""
    if rank != 0:
        return

    plt.figure(figsize=(10, 6))
    iterations = np.arange(1, len(objective_values) + 1)
    plt.plot(iterations, objective_values, marker='o', linestyle='-', markersize=4)
    plt.title('LoRA DCT Objective Values')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.grid(True, alpha=0.3)

    param_text = (
        f"Model: {args.model_name.split('/')[-1]}\n"
        f"Source Layers: {args.source_layer_start}-{args.source_layer_end}\n"
        f"Target Layer: {args.target_layer}\n"
        f"Num Factors: {args.num_factors}\n"
        f"LoRA Rank: {args.lora_rank}\n"
        f"Target Modules: {args.target_modules}"
    )
    plt.annotate(param_text, xy=(0.02, 0.97), xycoords='axes fraction',
                 va='top', ha='left', fontsize=9,
                 bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    plt_path = os.path.join(output_dir, f"{prefix}_objective_plot.png")
    plt.savefig(plt_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved objective plot to {plt_path}")

    # Log to wandb
    if use_wandb:
        wandb.log({'objective_plot': wandb.Image(plt_path)})


def plot_gram_matrix_density(flat_params, output_dir, prefix, rank, max_samples=100000, use_wandb=False):
    """
    Plot density of off-diagonal elements in the Gram matrix of flattened LoRA params.
    """
    if rank != 0:
        return

    # flat_params: (K, D)
    # Normalize each row
    flat_norm = flat_params / (flat_params.norm(dim=1, keepdim=True) + 1e-8)

    # Compute Gram matrix
    gram = flat_norm @ flat_norm.T  # (K, K)

    # Get off-diagonal elements
    mask = ~torch.eye(gram.shape[0], dtype=torch.bool, device=flat_params.device)
    off_diag = gram[mask]

    # Subsample if needed
    if off_diag.numel() > max_samples:
        indices = torch.randperm(off_diag.numel())[:max_samples]
        off_diag = off_diag[indices]

    off_diag_abs = off_diag.abs().detach().float().cpu().numpy()

    plt.figure(figsize=(10, 6))
    sns.kdeplot(off_diag_abs, fill=True, cut=0)
    plt.title('Density of |Off-Diagonal Elements| in LoRA Param Gram Matrix')
    plt.xlabel('Absolute Cosine Similarity')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)

    stats_text = (
        f"Mean: {off_diag_abs.mean():.4f}\n"
        f"Median: {np.median(off_diag_abs):.4f}\n"
        f"Std Dev: {off_diag_abs.std():.4f}\n"
        f"Max: {off_diag_abs.max():.4f}"
    )
    plt.annotate(stats_text, xy=(0.02, 0.97), xycoords='axes fraction',
                 va='top', ha='left', fontsize=9,
                 bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    plot_path = os.path.join(output_dir, f"{prefix}_gram_density.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Gram matrix density plot to {plot_path}")

    # Log to wandb
    if use_wandb:
        wandb.log({
            'gram_density_plot': wandb.Image(plot_path),
            'gram_cosine_sim_mean': float(off_diag_abs.mean()),
            'gram_cosine_sim_max': float(off_diag_abs.max()),
        })


def gather_tensors_to_rank0(local_tensor, rank, world_size, group=None):
    """Gather tensors from all ranks to rank 0."""
    if rank == 0:
        gathered = [torch.zeros_like(local_tensor) for _ in range(world_size)]
        dist.gather(local_tensor.contiguous(), gathered, dst=0, group=group)
        return torch.cat(gathered, dim=0)
    else:
        dist.gather(local_tensor.contiguous(), dst=0, group=group)
        return None


def _compute_layer_device_map(
    model_name: str,
    source_layer_start: int,
    target_layer: int,
    gpu_ids: list,
    full_model: bool = False,
):
    """Compute a device_map that distributes model layers across given GPUs.

    For trimmed model training: distributes layers [source_layer_start, target_layer]
    across gpu_ids. Embedding/head go on first GPU.

    For activation extraction (full_model=True): distributes all layers across gpu_ids.

    Returns a device_map dict for AutoModelForCausalLM.from_pretrained().
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.num_hidden_layers

    if full_model:
        layer_start = 0
        layer_end = num_layers - 1
    else:
        layer_start = source_layer_start
        layer_end = target_layer

    num_active_layers = layer_end - layer_start + 1
    num_gpus = len(gpu_ids)
    layers_per_gpu = (num_active_layers + num_gpus - 1) // num_gpus

    device_map = {}

    # Embedding, norm, and head placement.
    # lm_head must be on same GPU as embed_tokens (many models tie these weights).
    device_map["model.embed_tokens"] = gpu_ids[0]
    device_map["lm_head"] = gpu_ids[0]
    device_map["model.norm"] = gpu_ids[-1]
    device_map["model.rotary_emb"] = gpu_ids[0]

    # Distribute layers
    for i in range(num_layers):
        if full_model or (layer_start <= i <= layer_end):
            relative_idx = i - layer_start
            gpu_idx = min(relative_idx // layers_per_gpu, num_gpus - 1)
            device_map[f"model.layers.{i}"] = gpu_ids[gpu_idx]
        else:
            # For non-full model loading, layers outside range go to CPU
            # (they'll be None after trimming anyway)
            device_map[f"model.layers.{i}"] = "cpu"

    return device_map


def _extract_activations_distributed(
    args, rank, world_size, input_ids_list, device,
):
    """Extract activations. When training_tp_size > 1, rank 0 uses device_map='auto'.

    Returns (X, Y) lists of CPU tensors.
    """
    from transformers import AutoModelForCausalLM

    tp_size = args.training_tp_size
    # Optional attn implementation override (e.g. "flex_attention" / "eager").
    # gpt-oss needs eager or flex_attention for correct sinks+sliding; flex is
    # fused/fast. Used for BOTH activation (Y) extraction and the sliced training
    # forward so they stay consistent.
    _impl = os.environ.get("LORA_DCT_ATTN_IMPL", "").strip()
    attn_kw = {"attn_implementation": _impl} if _impl else {}

    if tp_size <= 1:
        # Original path: each rank loads full model on its GPU
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            **attn_kw,
        )
        for p in model.parameters():
            p.requires_grad = False

        X, Y = [], []
        for input_ids in tqdm(input_ids_list, disable=(rank != 0), desc="Forward pass"):
            input_ids_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_ids_tensor, output_hidden_states=True)
                X.append(outputs.hidden_states[args.source_layer_start].squeeze(0).cpu())
                Y.append(outputs.hidden_states[args.target_layer + 1].squeeze(0).cpu())

        return model, X, Y

    # TP mode: rank 0 extracts activations with device_map="auto", saves to disk
    activation_dir = os.path.join(args.output_dir, "_activations")
    x_path = os.path.join(activation_dir, "X.pt")
    y_path = os.path.join(activation_dir, "Y.pt")

    if rank == 0:
        os.makedirs(activation_dir, exist_ok=True)
        print("Loading model with device_map='auto' for activation extraction...")

        all_gpu_ids = list(range(torch.cuda.device_count()))
        dev_map = _compute_layer_device_map(
            args.model_name, 0, 0, all_gpu_ids, full_model=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map=dev_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            **attn_kw,
        )
        for p in model.parameters():
            p.requires_grad = False

        X, Y = [], []
        for input_ids in tqdm(input_ids_list, desc="Extracting activations"):
            # Find which device the embedding is on
            first_device = next(model.model.embed_tokens.parameters()).device
            input_ids_tensor = torch.tensor(input_ids, device=first_device).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_ids_tensor, output_hidden_states=True)
                X.append(outputs.hidden_states[args.source_layer_start].squeeze(0).cpu())
                Y.append(outputs.hidden_states[args.target_layer + 1].squeeze(0).cpu())

        torch.save(X, x_path)
        torch.save(Y, y_path)
        print(f"Saved activations to {activation_dir}")

        # Free the full model
        del model
        gc.collect()
        torch.cuda.empty_cache()

    dist.barrier()

    # All ranks load activations from disk
    if rank != 0:
        X = torch.load(x_path, weights_only=True)
        Y = torch.load(y_path, weights_only=True)
    else:
        pass  # rank 0 already has X, Y

    # Load trimmed model with per-group device_map
    num_groups = world_size // tp_size
    group_id = rank // tp_size
    group_gpus = [group_id * tp_size + i for i in range(tp_size)]

    if rank == 0:
        print(f"Loading trimmed model for training: {num_groups} groups of {tp_size} GPUs")

    dev_map = _compute_layer_device_map(
        args.model_name, args.source_layer_start, args.target_layer, group_gpus,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=dev_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    for p in model.parameters():
        p.requires_grad = False

    # Trim unused layers (set to None to free CPU memory for layers loaded to CPU)
    trim_model_for_training(model, args.source_layer_start, args.target_layer, rank)

    return model, X, Y


def main():
    start_time = time.time()

    # Setup distributed
    rank, world_size = setup_distributed()
    
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Set device
    device = f'cuda:{rank}'
    torch.cuda.set_device(device)
    
    # Parse target modules
    target_modules = [m.strip() for m in args.target_modules.split(',')]
    
    # Parse token indices
    token_idxs = parse_token_idxs(args.token_idxs)
    
    # Validate num_factors is divisible by effective parallelism
    tp_size = args.training_tp_size
    factor_world = world_size // tp_size if tp_size > 1 else world_size
    if args.num_factors % factor_world != 0:
        if rank == 0:
            old_num = args.num_factors
            args.num_factors = ((args.num_factors + factor_world - 1) // factor_world) * factor_world
            print(f"Adjusted num_factors from {old_num} to {args.num_factors} for divisibility")

    if rank == 0:
        print(f"="*60)
        print("LoRA DCT Distributed Training")
        print(f"="*60)
        print(f"Model: {args.model_name}")
        print(f"World size: {world_size}")
        if tp_size > 1:
            print(f"Training TP size: {tp_size} ({factor_world} factor-parallel groups)")
        print(f"Factors per group: {args.num_factors // factor_world}")
        print(f"Source layers: {args.source_layer_start}-{args.source_layer_end}")
        print(f"Target layer: {args.target_layer}")
        print(f"Target modules: {target_modules}")
        print(f"LoRA rank: {args.lora_rank}")
        print(f"="*60)

    # Initialize wandb (rank 0 only)
    use_wandb = args.use_wandb and HAS_WANDB and rank == 0
    if args.use_wandb and not HAS_WANDB and rank == 0:
        print("Warning: --use_wandb specified but wandb not installed. Skipping.")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                'model_name': args.model_name,
                'source_layers': f"{args.source_layer_start}-{args.source_layer_end}",
                'target_layer': args.target_layer,
                'target_modules': target_modules,
                'lora_rank': args.lora_rank,
                'num_factors': args.num_factors,
                'num_iters': args.num_iters,
                'beta': args.beta,
                'deflation': args.deflate,
                'deflation_temp': args.deflation_temp,
                'deflation_iterations': args.deflation_iterations,
                'norm_value': args.norm_value,
                'num_samples': args.num_samples,
                'world_size': world_size,
                'seed': args.seed,
                'separate_u': args.separate_u,
                'init': args.init,
            }
        )

    # Create output directory
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    dist.barrier()
    
    # Load dataset
    if args.split:
        dataset = load_from_disk(os.path.join(args.dataset, args.split))
    else:
        try:
            dataset = load_from_disk(args.dataset)
        except:
            dataset = load_dataset(args.dataset, split='train')
    
    # Select samples
    dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    if args.data_start is not None:
        dataset = dataset.select(range(args.data_start, args.data_end or len(dataset)))
    
    instructions = dataset[args.field]
    
    if rank == 0:
        print(f"Loaded {len(instructions)} instructions")
    
    # Load tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_name = args.tokenizer_name or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        padding_side="left",
        truncation_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare chat templates
    if args.system_prompt:
        chat_init = [{'content': args.system_prompt, 'role': 'system'}]
    else:
        chat_init = []

    chats = [chat_init + [{'content': content, 'role': 'user'}] for content in instructions]
    template_kwargs = {'enable_thinking': args.enable_thinking}
    if args.reasoning_effort is not None:
        template_kwargs['reasoning_effort'] = args.reasoning_effort
    examples = [tokenizer.apply_chat_template(
        chat, add_special_tokens=False, tokenize=False, add_generation_prompt=True,
        **template_kwargs,
    ) for chat in chats]

    # Tokenize each example individually (no padding, variable lengths)
    input_ids_list = []
    for ex in examples:
        tokens = tokenizer.encode(ex, truncation=True, max_length=args.max_length)
        input_ids_list.append(tokens)

    # Limit to num_samples
    num_samples = min(args.num_samples, len(input_ids_list))
    input_ids_list = input_ids_list[:num_samples]

    # Log sequence length stats
    seq_lengths = [len(ids) for ids in input_ids_list]
    if rank == 0:
        print(f"Sequence lengths: min={min(seq_lengths)}, max={max(seq_lengths)}, "
              f"mean={sum(seq_lengths)/len(seq_lengths):.0f}")

    # Load model and extract activations (handles both TP=1 and TP>1 paths)
    if rank == 0:
        print("Computing source and target activations...")

    model, X, Y = _extract_activations_distributed(
        args, rank, world_size, input_ids_list, device,
    )

    # For TP=1, trim model here (TP>1 path trims inside _extract_activations_distributed)
    if args.training_tp_size <= 1:
        trim_model_for_training(model, args.source_layer_start, args.target_layer, rank)

    d_model = model.config.hidden_size

    if rank == 0:
        print(f"Loaded {len(X)} samples with variable sequence lengths")

    # Create LoRA config
    config = LoRAFactorConfig(
        source_layers=(args.source_layer_start, args.source_layer_end),
        target_layer=args.target_layer,
        target_modules=target_modules,
        lora_rank=args.lora_rank,
        norm_value=args.norm_value,
        target_position_indices=token_idxs,
    )
    
    # Create and train LoRA DCT
    # With training_tp_size > 1, only group-leader ranks (rank % tp_size == 0)
    # participate in training. Each leader's model spans its TP group's GPUs
    # via device_map pipeline parallelism. A sub-process group is created for
    # the leaders so NCCL collectives work correctly.
    tp_size = args.training_tp_size
    if tp_size > 1:
        factor_rank = rank // tp_size
        factor_world = world_size // tp_size
        # Create sub-group for factor-parallel leaders
        leader_ranks = [i * tp_size for i in range(factor_world)]
        factor_group = dist.new_group(ranks=leader_ranks)
        is_leader = (rank % tp_size == 0)
    else:
        factor_rank = rank
        factor_world = world_size
        factor_group = None  # use default global group
        is_leader = True

    lora_dct = DistributedLoRADCT(
        num_factors=args.num_factors,
        rank=factor_rank,
        world_size=factor_world,
        config=config,
        process_group=factor_group,
    )
    
    # Only leader ranks participate in training (non-leaders wait at barrier below)
    if is_leader:
        # Create wandb callback for real-time logging
        def wandb_callback(iteration: int, objective_value: float):
            if use_wandb:
                wandb.log({'iteration': iteration, 'objective_value': objective_value})

        lora_factors, U_local = lora_dct.fit(
            model=model,
            X=X,  # List of (S_i, D) tensors
            Y=Y,  # List of (S_i, D) tensors
            factor_batch_size=args.factor_batch_size,
            init=args.init,
            norm_value=args.norm_value,
            max_iters=args.num_iters,
            beta=args.beta,
            deflation=args.deflate,
            soft_ortho_temp=args.deflation_temp,
            soft_ortho_iterations=args.deflation_iterations,
            separate_u=args.separate_u,
            callback=wandb_callback if use_wandb else None,
            offload_factors=not args.no_offload_factors,
            compile=args.compile
        )

        # Save results
        lora_dct.save(args.output_dir, args.output_prefix)

    # Synchronize all ranks (including non-leaders that skipped training)
    dist.barrier()

    # Save plots (rank 0 is always a leader)
    if rank == 0:
        save_objective_plot(
            lora_dct.objective_values,
            args.output_dir,
            args.output_prefix,
            args,
            rank,
            use_wandb=use_wandb
        )

    # Gather and plot gram matrix (leaders only)
    if is_leader:
        local_flat = lora_factors.get_flattened_all()
        all_flat = gather_tensors_to_rank0(local_flat, factor_rank, factor_world, group=factor_group)

        if factor_rank == 0 and all_flat is not None:
            plot_gram_matrix_density(all_flat.cpu(), args.output_dir, args.output_prefix, rank, use_wandb=use_wandb)

        # Log final scores to wandb
        if use_wandb:
            scores = lora_dct.scores
            if scores is not None:
                wandb.log({
                    'final_scores_mean': scores.mean().item(),
                    'final_scores_max': scores.max().item(),
                    'final_scores_min': scores.min().item(),
                    'final_scores_std': scores.std().item(),
                })
    
    # Save metadata
    if rank == 0:
        elapsed = time.time() - start_time
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)

        metadata = {
            'model_name': args.model_name,
            'source_layers': [args.source_layer_start, args.source_layer_end],
            'target_layer': args.target_layer,
            'target_modules': target_modules,
            'lora_rank': args.lora_rank,
            'num_factors': args.num_factors,
            'num_iters': args.num_iters,
            'beta': args.beta,
            'deflation': args.deflate,
            'deflation_temp': args.deflation_temp,
            'deflation_iterations': args.deflation_iterations,
            'norm_value': args.norm_value,
            'num_samples': num_samples,
            'seq_len_min': min(seq_lengths),
            'seq_len_max': max(seq_lengths),
            'seq_len_mean': sum(seq_lengths) / len(seq_lengths),
            'world_size': world_size,
            'seed': args.seed,
            'separate_u': args.separate_u,
            'init': args.init,
            'elapsed_time_seconds': elapsed,
            'elapsed_time_formatted': f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d}",
        }
        
        with open(os.path.join(args.output_dir, f"{args.output_prefix}_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"Saved {args.num_factors} LoRA factors to {args.output_dir}")
        print(f"{'='*60}")

    # Finish wandb logging
    if use_wandb:
        wandb.finish()

    # Cleanup
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

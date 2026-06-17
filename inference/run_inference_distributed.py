#!/usr/bin/env python
"""
Distributed LoRA DCT Inference with Ray + vLLM

Run inference with trained LoRA DCT factors using Ray for multi-GPU parallelism.
Each GPU worker handles a subset of factors with all prompts.

Usage:
    # Factor mode (from training results):
    python run_inference_distributed.py \
        --training_dir ./lora/outputs \
        --dataset ./data/prompts \
        --num_workers 4

    # Adapter dirs mode (from pre-built PEFT adapters):
    python run_inference_distributed.py \
        --adapter_dirs ./outputs/mce_run/composition/adapters \
        --dataset ./data/prompts \
        --model_name Qwen/Qwen3-8B \
        --num_workers 4

Requires:
    pip install ray vllm
"""

import os
import sys
import json
import time
import argparse
import tempfile
from typing import Dict, List, Any, Optional
from collections import defaultdict

import torch
import ray
from datasets import load_from_disk, load_dataset

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lora.peft_export import (
    load_lora_dct_results,
    export_factor_to_peft_dir,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run distributed inference with LoRA DCT factors using Ray + vLLM'
    )

    # Source arguments (mutually exclusive: training_dir vs adapter_dirs)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--training_dir', type=str, default=None,
                        help='Path to training outputs directory')
    source_group.add_argument('--adapter_dirs', type=str, default=None,
                        help='Path to directory containing pre-built PEFT adapter subdirectories')

    # Required
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset (HuggingFace or local)')

    # Optional training config
    parser.add_argument('--prefix', type=str, default='lora_dct',
                        help='Output prefix used during training')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name (reads from metadata if not provided)')

    # Dataset config
    parser.add_argument('--field', type=str, default='prompt',
                        help='Field name for prompts in dataset')
    parser.add_argument('--split', type=str, default=None,
                        help='Dataset split to use')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Max number of samples to use')

    # Generation config
    parser.add_argument('--max_tokens', type=int, default=256,
                        help='Maximum tokens to generate per request')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Nucleus sampling top-p')
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help='Repetition penalty (1.0 = no effect)')
    parser.add_argument('--top_k_factors', type=int, default=None,
                        help='Only use top K factors by score (default: all)')
    parser.add_argument('--factor_indices_file', type=str, default=None,
                        help='JSON file containing a list of factor indices to '
                             'evaluate (overrides --top_k_factors). Use for '
                             'evaluating an arbitrary subset (e.g. pilot-filter '
                             'survivors).')

    # vLLM config (per worker)
    parser.add_argument('--max_loras', type=int, default=4,
                        help='Max concurrent LoRAs in GPU memory per worker')
    parser.add_argument('--max_cpu_loras', type=int, default=None,
                        help='Max LoRAs cached in CPU RAM per worker (default: '
                             'per-worker adapter count, so every adapter stays '
                             'CPU-resident after first load and GPU swap-in is '
                             'a memcpy, not a disk read)')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                        help='GPU memory utilization for vLLM')
    parser.add_argument('--enforce_eager', action='store_true',
                        help='Disable CUDA graph and use eager mode')
    parser.add_argument('--max_num_seqs', type=int, default=64,
                        help='Maximum number of sequences per iteration (default: 64)')
    parser.add_argument('--max_model_len', type=int, default=None,
                        help='Maximum model context length for vLLM (default: use model config)')
    parser.add_argument('--batch_order', type=str, default='prompt_major',
                        choices=['prompt_major', 'adapter_major'],
                        help='Batch ordering: prompt_major groups adapters per prompt, '
                             'adapter_major groups prompts per adapter (default: prompt_major)')

    # Ray/distributed config
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help='GPUs per worker (vLLM TP). TP>1 gives more KV headroom for 70B; '
                             'num_workers defaults to (visible GPUs // TP).')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of GPU workers (default: auto-detect available GPUs)')
    parser.add_argument('--ray_address', type=str, default=None,
                        help='Ray cluster address (default: start local cluster)')

    # Prompt config
    parser.add_argument('--system_prompt', type=str, default=None,
                        help='System prompt (reads from metadata if not provided)')

    # Baseline config
    parser.add_argument('--include_baseline', action='store_true',
                        help='Also run baseline (no adapter) generation on worker 0')

    # Chat template config
    parser.add_argument('--enable_thinking', action='store_true',
                        help='Enable thinking/reasoning in chat template (for supported models)')
    parser.add_argument('--reasoning_effort', type=str, default=None,
                        choices=[None, 'low', 'medium', 'high'],
                        help="gpt-oss reasoning effort (low/medium/high). If unset, model's chat-template default applies (medium for gpt-oss).")

    # Output config
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output JSON path (default: {training_dir}/inference_results_distributed.json)')

    return parser.parse_args()


# === Shared Utilities ===

def load_metadata(training_dir: str, prefix: str) -> Dict[str, Any]:
    """Load training metadata."""
    metadata_path = os.path.join(training_dir, f"{prefix}_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}


def get_top_k_factors(scores: torch.Tensor, k: Optional[int]) -> List[int]:
    """Get indices of top-k factors by score."""
    if k is None or k >= len(scores):
        return torch.argsort(scores, descending=True).tolist()
    return torch.argsort(scores, descending=True)[:k].tolist()


def prepare_chat_prompts(
    instructions: List[str],
    tokenizer,
    system_prompt: str,
    enable_thinking: bool = False,
    reasoning_effort: Optional[str] = None,
) -> List[str]:
    """Prepare prompts using chat template.

    `reasoning_effort` is forwarded to gpt-oss chat templates (low/medium/high);
    other templates silently ignore unknown kwargs."""
    prompts = []
    template_kwargs = {'enable_thinking': enable_thinking}
    if reasoning_effort is not None:
        template_kwargs['reasoning_effort'] = reasoning_effort

    for instruction in instructions:
        if system_prompt:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': instruction}
            ]
        else:
            messages = [{'role': 'user', 'content': instruction}]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        prompts.append(prompt)

    return prompts


def filter_overlength_prompts(prompts, raw_prompts, tokenizer, max_model_len):
    """Drop prompts whose token count exceeds max_model_len. Returns filtered lists and count dropped."""
    if max_model_len is None:
        return prompts, raw_prompts, 0
    token_lengths = [len(tokenizer.encode(p)) for p in prompts]
    valid = [i for i, l in enumerate(token_lengths) if l <= max_model_len]
    num_dropped = len(prompts) - len(valid)
    if num_dropped > 0:
        print(f"WARNING: Dropping {num_dropped}/{len(prompts)} prompts exceeding max_model_len={max_model_len}")
    return [prompts[i] for i in valid], [raw_prompts[i] for i in valid], num_dropped


def save_results(
    output_path: str,
    metadata: Dict[str, Any],
    results: List[Dict]
):
    """Save results to JSON."""
    output = {
        'metadata': metadata,
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved results to {output_path}")


def partition_factors(factor_indices: List[int], num_workers: int) -> List[List[int]]:
    """
    Partition factors across workers using round-robin for balanced load.

    Args:
        factor_indices: List of factor indices (sorted by score)
        num_workers: Number of worker processes

    Returns:
        List of lists, where assignments[i] contains factor indices for worker i
    """
    assignments = [[] for _ in range(num_workers)]

    for idx, factor_idx in enumerate(factor_indices):
        worker_id = idx % num_workers
        assignments[worker_id].append(factor_idx)

    return assignments


def scan_adapter_dirs(adapter_dirs_path: str) -> Dict[str, str]:
    """
    Scan a directory for pre-built PEFT adapter subdirectories.

    Returns dict mapping adapter_name -> adapter_path, sorted by name.
    """
    adapters = {}
    for entry in sorted(os.listdir(adapter_dirs_path)):
        subdir = os.path.join(adapter_dirs_path, entry)
        config_path = os.path.join(subdir, "adapter_config.json")
        if os.path.isdir(subdir) and os.path.exists(config_path):
            adapters[entry] = subdir
    return adapters


def _round_up_to_valid_vllm_rank(rank: int) -> int:
    """Round up to the next valid vLLM max_lora_rank value.

    NOTE: rank=1 is technically a valid LoRA rank but vLLM's Punica BF16 path
    asserts 16-byte aligned strides, which requires rank ≥ 8 for BF16 (2 B
    per element × 8 = 16 B). Quantized (MXFP4) paths via Marlin do support
    rank=1. To be safe we pad up to 8 unconditionally — vLLM allocates rank-8
    slots and the PEFT adapter's actual rank-1 weights load zero-padded.
    """
    valid_ranks = [8, 16, 32, 64, 128, 256, 320, 512]
    for v in valid_ranks:
        if v >= rank:
            return v
    return valid_ranks[-1]


def get_max_lora_rank_from_adapters(adapter_paths: Dict[str, str]) -> int:
    """Read adapter_config.json from each adapter dir and return the max rank (rounded up to valid vLLM value)."""
    max_rank = 0
    for name, path in adapter_paths.items():
        config_path = os.path.join(path, "adapter_config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        max_rank = max(max_rank, config.get('r', 1))
    return _round_up_to_valid_vllm_rank(max_rank)


# === Ray Actor ===

@ray.remote(num_gpus=1)
class VLLMInferenceWorker:
    """
    Ray Actor that owns a vLLM instance and processes a subset of factors.

    Each worker:
    1. Initializes vLLM once with the base model
    2. Receives factor indices to process
    3. Exports those factors to PEFT directories
    4. Runs inference for all prompts with each assigned factor
    5. Returns results
    """

    def __init__(
        self,
        worker_id: int,
        model_name: str,
        lora_rank: int,
        max_loras: int,
        max_num_seqs: int,
        gpu_memory_utilization: float,
        enforce_eager: bool,
        max_model_len: Optional[int] = None,
        max_cpu_loras: Optional[int] = None,
        tensor_parallel_size: int = 1,
    ):
        """Initialize the vLLM engine once."""
        self.worker_id = worker_id
        self.model_name = model_name

        # Import vLLM inside actor (GPU-specific initialization)
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        self.SamplingParams = SamplingParams
        self.LoRARequest = LoRARequest

        print(f"Worker {worker_id}: Initializing vLLM with model {model_name}")

        # Initialize vLLM engine (expensive, done once)
        # NOTE: force FLASH_ATTN attention backend on B200 (sm_100a) — the
        # flashinfer JIT path needs curand/cublasLt/nvrtc dev headers which
        # are not installed on this box. The backend is forced via the
        # VLLM_ATTENTION_BACKEND env var set in the Ray runtime_env; we do NOT
        # also pass the attention_backend kwarg here because vLLM >=0.13 treats
        # the env var and the kwarg as mutually exclusive and raises if both set.
        llm_kwargs = dict(
            model=model_name,
            enable_lora=True,
            max_loras=max_loras,
            max_lora_rank=lora_rank,
            max_num_seqs=max_num_seqs,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=enforce_eager,
            enable_prefix_caching=True,
        )
        # vLLM internal TP within a Ray actor: use the mp backend (the actor owns
        # tensor_parallel_size GPUs; let vLLM spawn its own TP workers via mp).
        if tensor_parallel_size > 1:
            llm_kwargs['distributed_executor_backend'] = 'mp'
        if max_cpu_loras is not None:
            llm_kwargs['max_cpu_loras'] = max_cpu_loras
        if max_model_len is not None:
            llm_kwargs['max_model_len'] = max_model_len
        # Patch vLLM's LoRA ops to tolerate non-contiguous inputs (gpt-oss). Takes
        # effect when the engine is in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0,
        # set in the runtime_env for sink models).
        _patch_vllm_lora_contiguous()
        self.llm = LLM(**llm_kwargs)

        self.tokenizer = self.llm.get_tokenizer()
        print(f"Worker {worker_id}: vLLM initialized successfully")

    def shutdown(self):
        """Explicitly shut down the vLLM engine to release GPU memory."""
        del self.llm
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Worker {self.worker_id}: vLLM engine shut down")

    def process_factors(
        self,
        factor_indices: List[int],
        all_factors: torch.Tensor,
        config: Dict[str, Any],
        prompts: List[str],
        raw_prompts: List[str],
        sampling_params_dict: Dict[str, Any],
        scores: torch.Tensor,
        batch_order: str = 'prompt_major',
    ) -> List[Dict]:
        """
        Process a batch of factors with all prompts.

        Args:
            factor_indices: Which factors this worker should process
            all_factors: Tensor of all flattened LoRA parameters
            config: Training config dict
            prompts: Tokenized/templated prompts
            raw_prompts: Original prompt strings
            sampling_params_dict: vLLM sampling parameters
            scores: Factor scores for metadata
            batch_order: 'prompt_major' or 'adapter_major' loop ordering

        Returns:
            List of result dicts, one per factor
        """
        print(f"Worker {self.worker_id}: Processing {len(factor_indices)} factors ({batch_order})")

        # Create temp directory for this worker's factors
        with tempfile.TemporaryDirectory() as lora_dir:
            # Build results dict to pass to export function
            results_for_export = {
                'all_factors': all_factors,
                'config': config,
            }

            # Export assigned factors to PEFT directories
            lora_paths = {}
            for idx in factor_indices:
                lora_paths[idx] = export_factor_to_peft_dir(
                    idx, results_for_export, lora_dir, self.model_name
                )

            # Pre-build LoRARequest objects
            lora_requests = {}
            for factor_idx in factor_indices:
                lora_requests[factor_idx] = self.LoRARequest(
                    lora_name=f"factor_{factor_idx}",
                    lora_int_id=factor_idx + 1,  # vLLM requires int_id > 0
                    lora_path=lora_paths[factor_idx],
                )

            # Build requests for all (factor, prompt) combinations
            all_prompts_batch = []
            all_lora_requests = []
            request_metadata = []

            if batch_order == 'prompt_major':
                for prompt_idx, prompt in enumerate(prompts):
                    for factor_idx in factor_indices:
                        all_prompts_batch.append(prompt)
                        all_lora_requests.append(lora_requests[factor_idx])
                        request_metadata.append({
                            'factor_idx': factor_idx,
                            'prompt_idx': prompt_idx,
                        })
            else:  # adapter_major
                for factor_idx in factor_indices:
                    for prompt_idx, prompt in enumerate(prompts):
                        all_prompts_batch.append(prompt)
                        all_lora_requests.append(lora_requests[factor_idx])
                        request_metadata.append({
                            'factor_idx': factor_idx,
                            'prompt_idx': prompt_idx,
                        })

            # Create sampling params
            sampling_params = self.SamplingParams(**sampling_params_dict)

            print(f"Worker {self.worker_id}: Running inference on {len(all_prompts_batch)} requests")

            # Run batch inference
            outputs = self.llm.generate(
                all_prompts_batch,
                sampling_params,
                lora_request=all_lora_requests,
            )

            # Organize results by factor
            results = self._organize_results(
                outputs, request_metadata, scores, raw_prompts, factor_indices
            )

        print(f"Worker {self.worker_id}: Completed processing")
        return results

    def process_adapters(
        self,
        adapter_assignments: List[Dict[str, Any]],
        prompts: List[str],
        raw_prompts: List[str],
        sampling_params_dict: Dict[str, Any],
        batch_order: str = 'prompt_major',
    ) -> List[Dict]:
        """
        Process pre-built PEFT adapter directories with all prompts.

        Args:
            adapter_assignments: List of dicts with 'name', 'path', 'int_id' keys
            prompts: Tokenized/templated prompts
            raw_prompts: Original prompt strings
            sampling_params_dict: vLLM sampling parameters
            batch_order: 'prompt_major' or 'adapter_major' loop ordering

        Returns:
            List of result dicts, one per adapter
        """
        print(f"Worker {self.worker_id}: Processing {len(adapter_assignments)} adapters ({batch_order})")

        # Pre-build LoRARequest objects
        lora_requests = {}
        for adapter in adapter_assignments:
            lora_requests[adapter['name']] = self.LoRARequest(
                lora_name=adapter['name'],
                lora_int_id=adapter['int_id'],
                lora_path=adapter['path'],
            )

        all_prompts_batch = []
        all_lora_requests = []
        request_metadata = []

        if batch_order == 'prompt_major':
            for prompt_idx, prompt in enumerate(prompts):
                for adapter in adapter_assignments:
                    all_prompts_batch.append(prompt)
                    all_lora_requests.append(lora_requests[adapter['name']])
                    request_metadata.append({
                        'adapter_name': adapter['name'],
                        'prompt_idx': prompt_idx,
                    })
        else:  # adapter_major
            for adapter in adapter_assignments:
                for prompt_idx, prompt in enumerate(prompts):
                    all_prompts_batch.append(prompt)
                    all_lora_requests.append(lora_requests[adapter['name']])
                    request_metadata.append({
                        'adapter_name': adapter['name'],
                        'prompt_idx': prompt_idx,
                    })

        sampling_params = self.SamplingParams(**sampling_params_dict)

        print(f"Worker {self.worker_id}: Running inference on {len(all_prompts_batch)} requests")

        outputs = self.llm.generate(
            all_prompts_batch,
            sampling_params,
            lora_request=all_lora_requests,
        )

        # Organize results by adapter
        results_by_adapter = defaultdict(list)
        for output, req in zip(outputs, request_metadata):
            adapter_name = req['adapter_name']
            prompt_idx = req['prompt_idx']
            results_by_adapter[adapter_name].append({
                'prompt_idx': prompt_idx,
                'prompt': raw_prompts[prompt_idx],
                'response': output.outputs[0].text,
            })

        results = []
        for adapter in adapter_assignments:
            results.append({
                'adapter_name': adapter['name'],
                'factor_idx': adapter.get('int_id', 0),
                'score': 0.0,
                'responses': results_by_adapter[adapter['name']],
            })

        print(f"Worker {self.worker_id}: Completed processing")
        return results

    def process_prompt_subset(
        self,
        prompt_indices: List[int],
        all_prompts: List[str],
        all_raw_prompts: List[str],
        sampling_params_dict: Dict[str, Any],
        adapter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """
        Process a subset of prompts for a single adapter or baseline.

        Used for prompt-parallel dispatch where each worker handles
        a portion of prompts rather than a portion of adapters.

        Args:
            prompt_indices: Global prompt indices this worker should process
            all_prompts: Full prompt list (from object store)
            all_raw_prompts: Full raw prompt list (from object store)
            sampling_params_dict: vLLM sampling parameters
            adapter: Dict with 'name', 'path', 'int_id' keys, or None for baseline

        Returns:
            Single-element list with result dict containing this subset's responses
        """
        prompts = [all_prompts[i] for i in prompt_indices]
        raw_prompts = [all_raw_prompts[i] for i in prompt_indices]

        sampling_params = self.SamplingParams(**sampling_params_dict)

        if adapter is not None:
            adapter_name = adapter['name']
            print(f"Worker {self.worker_id}: Processing {len(prompts)} prompts for adapter '{adapter_name}'")
            lora_request = self.LoRARequest(
                lora_name=adapter['name'],
                lora_int_id=adapter['int_id'],
                lora_path=adapter['path'],
            )
            outputs = self.llm.generate(prompts, sampling_params, lora_request=lora_request)
        else:
            adapter_name = 'baseline'
            print(f"Worker {self.worker_id}: Processing {len(prompts)} prompts for baseline")
            outputs = self.llm.generate(prompts, sampling_params)

        responses = []
        for local_idx, output in enumerate(outputs):
            responses.append({
                'prompt_idx': prompt_indices[local_idx],
                'prompt': raw_prompts[local_idx],
                'response': output.outputs[0].text,
            })

        print(f"Worker {self.worker_id}: Completed {adapter_name} ({len(prompts)} prompts)")
        return [{
            'adapter_name': adapter_name,
            'factor_idx': adapter.get('int_id', 0) if adapter else 0,
            'score': 0.0,
            'responses': responses,
        }]

    def _organize_results(
        self,
        outputs: List,
        request_metadata: List[Dict],
        scores: torch.Tensor,
        raw_prompts: List[str],
        factor_indices: List[int],
    ) -> List[Dict]:
        """Organize vLLM outputs by factor."""
        results_by_factor = defaultdict(list)

        for output, request in zip(outputs, request_metadata):
            factor_idx = request['factor_idx']
            prompt_idx = request['prompt_idx']
            results_by_factor[factor_idx].append({
                'prompt_idx': prompt_idx,
                'prompt': raw_prompts[prompt_idx],
                'response': output.outputs[0].text,
            })

        results = []
        for factor_idx in factor_indices:
            results.append({
                'factor_idx': factor_idx,
                'score': scores[factor_idx].item(),
                'responses': results_by_factor[factor_idx]
            })

        return results

    def health_check(self) -> Dict[str, Any]:
        """Return worker status for debugging."""
        return {
            'worker_id': self.worker_id,
            'model_name': self.model_name,
            'status': 'ready',
        }


# === Coordinator ===

def _model_needs_auto_backend(model_name) -> bool:
    """gpt-oss (attention sinks) / Gemma (softcap) can't use vLLM's FLASH_ATTN
    backend (it asserts no sinks). Let vLLM auto-select a sink-capable backend
    (FlashInfer/Triton) for them instead of forcing FLASH_ATTN."""
    if not model_name:
        return False
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        return False
    if getattr(cfg, "attn_logit_softcapping", None) is not None:
        return True
    return getattr(cfg, "model_type", "") in ("gpt_oss",)


def _patch_vllm_lora_contiguous():
    """vLLM 0.18.x's lora_shrink/lora_expand Triton ops assert contiguous inputs,
    but gpt-oss feeds a non-contiguous o_proj LoRA input (Qwen's is contiguous, so
    Qwen is unaffected). Wrap the ops to make the input contiguous. No-op for
    already-contiguous inputs. Must run in the engine process (we set
    VLLM_ENABLE_V1_MULTIPROCESSING=0 for sink models so the engine is in-process)."""
    try:
        import vllm.lora.punica_wrapper.punica_gpu as pg
    except Exception:
        return
    for name in ("lora_shrink", "lora_expand"):
        orig = getattr(pg, name, None)
        if orig is None or getattr(orig, "_contig_patched", False):
            continue

        def _make(o):
            def wrapped(inputs, *a, **k):
                try:
                    if not inputs.is_contiguous():
                        inputs = inputs.contiguous()
                except Exception:
                    pass
                return o(inputs, *a, **k)
            wrapped._contig_patched = True
            return wrapped

        setattr(pg, name, _make(orig))


def _vllm_runtime_env(model_name=None) -> dict:
    """Build the Ray runtime_env. Forces FLASH_ATTN (validated path) for standard
    models; for sink/softcap models, omits the force so vLLM auto-selects a
    sink-capable backend. An explicit VLLM_ATTENTION_BACKEND in the environment
    always wins."""
    env = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
    override = os.environ.get("VLLM_ATTENTION_BACKEND")
    if override:
        env["VLLM_ATTENTION_BACKEND"] = override
    elif not _model_needs_auto_backend(model_name):
        env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    # Sink/softcap models (gpt-oss) need the lora_shrink contiguity patch, which
    # must run in the engine process -> run the engine in-process (no V1 subproc).
    if _model_needs_auto_backend(model_name):
        env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # else: omit -> vLLM auto-selects (FlashInfer for gpt-oss sinks on SM100)
    if os.environ.get("CPATH"):
        env["CPATH"] = os.environ["CPATH"]
    # Pass through any VLLM_* env vars set in the launching process (e.g.
    # VLLM_USE_FLASHINFER_MOE_MXFP4_BF16 to route MoE off the Triton path).
    for k, v in os.environ.items():
        if k.startswith("VLLM_") and k not in env:
            env[k] = v
    return {"env_vars": env}


def run_distributed_inference(args) -> List[Dict]:
    """
    Main coordinator function that orchestrates distributed inference.
    """
    # Initialize Ray. Disable flashinfer JIT paths (attention + sampler) so vLLM
    # falls back to FLASH_ATTN + PyTorch-native sampling. The local CUDA install
    # is missing curand/cublasLt/nvrtc headers and can't JIT-compile flashinfer.
    runtime_env = _vllm_runtime_env(args.model_name)
    if args.ray_address:
        ray.init(address=args.ray_address, runtime_env=runtime_env)
    else:
        ray.init(runtime_env=runtime_env)

    # Determine number of workers
    if args.num_workers is None:
        num_workers = int(ray.available_resources().get('GPU', 1)) // max(1, getattr(args, 'tensor_parallel_size', 1))
    else:
        num_workers = args.num_workers

    print(f"Ray initialized with {num_workers} GPU workers")

    # Load training results on driver (CPU)
    print(f"Loading training results from {args.training_dir}")
    results = load_lora_dct_results(args.training_dir, args.prefix)
    config = results['config']
    scores = results['scores']
    all_factors = results['all_factors']

    # Load metadata
    metadata = load_metadata(args.training_dir, args.prefix)
    model_name = args.model_name or metadata.get('model_name')
    if not model_name:
        raise ValueError("Model name must be provided via --model_name or in training metadata")

    system_prompt = args.system_prompt or metadata.get('system_prompt', '')

    # Get factor indices
    if args.factor_indices_file is not None:
        with open(args.factor_indices_file) as _f:
            factor_indices = json.load(_f)
        if not isinstance(factor_indices, list):
            raise ValueError(f"{args.factor_indices_file} must contain a JSON list")
        factor_indices = [int(x) for x in factor_indices]
        print(f"Using {len(factor_indices)} factors from {args.factor_indices_file}")
    else:
        factor_indices = get_top_k_factors(scores, args.top_k_factors)
        print(f"Using {len(factor_indices)} factors across {num_workers} workers")

    # Load dataset
    print(f"Loading dataset from {args.dataset}")
    if os.path.exists(args.dataset):
        dataset = load_from_disk(args.dataset + (f"/{args.split}" if args.split else ""))
    else:
        dataset = load_dataset(args.dataset, split=args.split)

    if args.num_samples:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    raw_prompts = dataset[args.field]
    print(f"Loaded {len(raw_prompts)} prompts")

    # Load tokenizer for prompt preparation (lightweight, on driver)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    prompts = prepare_chat_prompts(raw_prompts, tokenizer, system_prompt, args.enable_thinking, args.reasoning_effort)
    prompts, raw_prompts, num_dropped = filter_overlength_prompts(prompts, raw_prompts, tokenizer, args.max_model_len)

    # Partition factors across workers
    factor_assignments = partition_factors(factor_indices, num_workers)
    for i, assignment in enumerate(factor_assignments):
        print(f"  Worker {i}: {len(assignment)} factors")

    # Default max_cpu_loras to max per-worker adapter count so every adapter
    # stays CPU-resident after first load (avoids re-reading from disk on each
    # swap-in when m >> max_loras).
    max_per_worker = max(len(a) for a in factor_assignments)
    eff_max_cpu_loras = args.max_cpu_loras if args.max_cpu_loras is not None else max_per_worker
    # vLLM requires max_cpu_loras >= max_loras; cap max_loras to the per-worker
    # factor count (when there are fewer factors/worker than the requested
    # max_loras, e.g. few-factor smoke runs).
    eff_max_loras = min(args.max_loras, max_per_worker)
    # Leave CUDA graphs / torch.compile ON for sink models (gpt-oss). The old
    # force-eager workaround was a vLLM 0.18 / torch 2.10 CUDAGraph fragility that
    # is gone on the pinned vLLM 0.21 stack (gpt-oss runs graphs fine, ~2x faster
    # inference). Only enforce eager when explicitly requested.
    eff_enforce_eager = args.enforce_eager

    # Create worker actors
    print("Creating worker actors...")
    _tp = getattr(args, 'tensor_parallel_size', 1)
    workers = [
        VLLMInferenceWorker.options(num_gpus=_tp).remote(
            worker_id=i,
            model_name=model_name,
            lora_rank=_round_up_to_valid_vllm_rank(config['lora_rank']),
            max_loras=eff_max_loras,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=eff_enforce_eager,
            max_model_len=args.max_model_len,
            max_cpu_loras=eff_max_cpu_loras,
            tensor_parallel_size=_tp,
        )
        for i in range(num_workers)
    ]

    # Wait for all workers to initialize
    print("Waiting for workers to initialize vLLM engines...")
    health_checks = ray.get([w.health_check.remote() for w in workers])
    for check in health_checks:
        print(f"  Worker {check['worker_id']}: {check['status']}")

    # Prepare sampling params
    sampling_params_dict = {
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'repetition_penalty': args.repetition_penalty,
    }

    # Put large data in Ray object store (shared memory)
    all_factors_ref = ray.put(all_factors)
    config_ref = ray.put(config)
    prompts_ref = ray.put(prompts)
    raw_prompts_ref = ray.put(raw_prompts)
    scores_ref = ray.put(scores)

    # Dispatch work to all workers in parallel
    print(f"Dispatching inference to {num_workers} workers... (batch_order={args.batch_order})")
    futures = [
        workers[i].process_factors.remote(
            factor_indices=factor_assignments[i],
            all_factors=all_factors_ref,
            config=config_ref,
            prompts=prompts_ref,
            raw_prompts=raw_prompts_ref,
            sampling_params_dict=sampling_params_dict,
            scores=scores_ref,
            batch_order=args.batch_order,
        )
        for i in range(num_workers)
    ]

    # Collect results from all workers
    print("Waiting for workers to complete...")
    all_results = ray.get(futures)

    # Merge results (flatten list of lists)
    merged_results = []
    for worker_results in all_results:
        merged_results.extend(worker_results)

    # Sort by factor index for consistent ordering
    merged_results.sort(key=lambda x: x['factor_idx'])

    # Gracefully shut down vLLM engines before tearing down Ray
    ray.get([w.shutdown.remote() for w in workers])
    ray.shutdown()

    return merged_results, {
        'model_name': model_name,
        'training_dir': args.training_dir,
        'dataset': args.dataset,
        'num_factors': len(factor_indices),
        'num_prompts': len(raw_prompts),
        'num_prompts_dropped': num_dropped,
        'num_workers': num_workers,
        'system_prompt': system_prompt,
        'generation_config': {
            'max_tokens': args.max_tokens,
            'temperature': args.temperature,
            'top_p': args.top_p,
        },
        'lora_config': {
            'lora_rank': config['lora_rank'],
            'source_layers': config['source_layers'],
            'target_modules': config['target_modules'],
        }
    }


def run_distributed_inference_adapter_dirs(args) -> List[Dict]:
    """
    Coordinator for --adapter_dirs mode: load pre-built PEFT adapters and run inference.
    """
    # Initialize Ray. Disable flashinfer JIT paths (attention + sampler) so vLLM
    # falls back to FLASH_ATTN + PyTorch-native sampling. The local CUDA install
    # is missing curand/cublasLt/nvrtc headers and can't JIT-compile flashinfer.
    runtime_env = _vllm_runtime_env(args.model_name)
    if args.ray_address:
        ray.init(address=args.ray_address, runtime_env=runtime_env)
    else:
        ray.init(runtime_env=runtime_env)

    # Determine number of workers
    if args.num_workers is None:
        num_workers = int(ray.available_resources().get('GPU', 1)) // max(1, getattr(args, 'tensor_parallel_size', 1))
    else:
        num_workers = args.num_workers

    print(f"Ray initialized with {num_workers} GPU workers")

    # Scan adapter directories
    print(f"Scanning adapter directories in {args.adapter_dirs}")
    adapter_map = scan_adapter_dirs(args.adapter_dirs)
    if not adapter_map:
        raise ValueError(f"No valid PEFT adapter directories found in {args.adapter_dirs}")
    print(f"Found {len(adapter_map)} adapters: {list(adapter_map.keys())}")

    # Determine max LoRA rank across all adapters
    max_lora_rank = get_max_lora_rank_from_adapters(adapter_map)
    print(f"Max LoRA rank across adapters: {max_lora_rank}")

    # Model name is required in adapter_dirs mode
    model_name = args.model_name
    if not model_name:
        raise ValueError("--model_name is required when using --adapter_dirs")

    system_prompt = args.system_prompt or ''

    # Load dataset
    print(f"Loading dataset from {args.dataset}")
    if os.path.exists(args.dataset):
        dataset = load_from_disk(args.dataset + (f"/{args.split}" if args.split else ""))
    else:
        dataset = load_dataset(args.dataset, split=args.split)

    if args.num_samples:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    raw_prompts = dataset[args.field]
    print(f"Loaded {len(raw_prompts)} prompts")

    # Load tokenizer for prompt preparation
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    prompts = prepare_chat_prompts(raw_prompts, tokenizer, system_prompt, args.enable_thinking, args.reasoning_effort)
    prompts, raw_prompts, num_dropped = filter_overlength_prompts(prompts, raw_prompts, tokenizer, args.max_model_len)

    # Build adapter list with int_ids (vLLM requires int_id > 0)
    adapter_list = []
    for int_id, (name, path) in enumerate(adapter_map.items(), start=1):
        adapter_list.append({
            'name': name,
            'path': path,
            'int_id': int_id,
        })

    include_baseline = getattr(args, 'include_baseline', False)

    if include_baseline:
        # Prompt-parallel mode: each worker handles all adapters on a prompt subset.
        # max_loras only needs to cover the adapter count (not partitioned).
        max_loras = min(args.max_loras, len(adapter_list))
    else:
        # Adapter-parallel mode: partition adapters across workers (round-robin)
        adapter_assignments = [[] for _ in range(num_workers)]
        for idx, adapter in enumerate(adapter_list):
            worker_id = idx % num_workers
            adapter_assignments[worker_id].append(adapter)

        for i, assignment in enumerate(adapter_assignments):
            names = [a['name'] for a in assignment]
            print(f"  Worker {i}: {len(assignment)} adapters ({names})")

        # Cap max_loras to the largest per-worker adapter count
        max_per_worker = max(len(a) for a in adapter_assignments)
        max_loras = min(args.max_loras, max_per_worker)

    # Default max_cpu_loras to max per-worker adapter count (keep every adapter
    # in CPU RAM so GPU swap-in is a memcpy, not a disk read).
    if include_baseline:
        adapter_pool_size = len(adapter_list)
    else:
        adapter_pool_size = max(len(a) for a in adapter_assignments)
    eff_max_cpu_loras = args.max_cpu_loras if args.max_cpu_loras is not None else adapter_pool_size

    # Create worker actors
    print("Creating worker actors...")
    _tp = getattr(args, 'tensor_parallel_size', 1)
    workers = [
        VLLMInferenceWorker.options(num_gpus=_tp).remote(
            worker_id=i,
            model_name=model_name,
            lora_rank=max_lora_rank,
            max_loras=max_loras,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            max_model_len=args.max_model_len,
            max_cpu_loras=eff_max_cpu_loras,
            tensor_parallel_size=_tp,
        )
        for i in range(num_workers)
    ]

    # Wait for all workers to initialize
    print("Waiting for workers to initialize vLLM engines...")
    health_checks = ray.get([w.health_check.remote() for w in workers])
    for check in health_checks:
        print(f"  Worker {check['worker_id']}: {check['status']}")

    # Prepare sampling params
    sampling_params_dict = {
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'repetition_penalty': args.repetition_penalty,
    }

    # Put shared data in Ray object store
    prompts_ref = ray.put(prompts)
    raw_prompts_ref = ray.put(raw_prompts)

    if include_baseline:
        # Prompt-parallel dispatch: split prompts across workers,
        # each worker processes all adapters + baseline on its subset.
        # Better than adapter-parallel when adapter count is small.
        prompt_indices = list(range(len(prompts)))
        prompt_assignments = partition_factors(prompt_indices, num_workers)

        print(f"Dispatching prompt-parallel inference to {num_workers} workers...")
        for i, assignment in enumerate(prompt_assignments):
            print(f"  Worker {i}: {len(assignment)} prompts")

        futures = []
        for i in range(num_workers):
            if not prompt_assignments[i]:
                continue
            # Baseline pass
            futures.append(
                workers[i].process_prompt_subset.remote(
                    prompt_indices=prompt_assignments[i],
                    all_prompts=prompts_ref,
                    all_raw_prompts=raw_prompts_ref,
                    sampling_params_dict=sampling_params_dict,
                    adapter=None,
                )
            )
            # Adapter passes
            for adapter in adapter_list:
                futures.append(
                    workers[i].process_prompt_subset.remote(
                        prompt_indices=prompt_assignments[i],
                        all_prompts=prompts_ref,
                        all_raw_prompts=raw_prompts_ref,
                        sampling_params_dict=sampling_params_dict,
                        adapter=adapter,
                    )
                )
    else:
        # Adapter-parallel dispatch: split adapters across workers,
        # each worker processes all prompts for its assigned adapters.
        print(f"Dispatching inference to {num_workers} workers... (batch_order={args.batch_order})")
        futures = [
            workers[i].process_adapters.remote(
                adapter_assignments=adapter_assignments[i],
                prompts=prompts_ref,
                raw_prompts=raw_prompts_ref,
                sampling_params_dict=sampling_params_dict,
                batch_order=args.batch_order,
            )
            for i in range(num_workers)
        ]

    # Collect results from all workers
    print("Waiting for workers to complete...")
    all_results = ray.get(futures)

    if include_baseline:
        # Merge prompt-parallel results: combine partial responses per adapter
        merged_by_name = {}
        for worker_results in all_results:
            for result in worker_results:
                name = result['adapter_name']
                if name not in merged_by_name:
                    merged_by_name[name] = {
                        'adapter_name': name,
                        'factor_idx': result.get('factor_idx', 0),
                        'score': result.get('score', 0.0),
                        'responses': [],
                    }
                merged_by_name[name]['responses'].extend(result['responses'])

        merged_results = []
        for entry in merged_by_name.values():
            entry['responses'].sort(key=lambda x: x['prompt_idx'])
            merged_results.append(entry)
    else:
        # Merge adapter-parallel results (flatten list of lists)
        merged_results = []
        for worker_results in all_results:
            merged_results.extend(worker_results)

    # Sort by adapter name for consistent ordering
    merged_results.sort(key=lambda x: x.get('adapter_name', ''))

    # Gracefully shut down vLLM engines before tearing down Ray
    ray.get([w.shutdown.remote() for w in workers])
    ray.shutdown()

    return merged_results, {
        'model_name': model_name,
        'adapter_dirs': args.adapter_dirs,
        'dataset': args.dataset,
        'num_adapters': len(adapter_map),
        'num_prompts': len(raw_prompts),
        'num_prompts_dropped': num_dropped,
        'num_workers': num_workers,
        'system_prompt': system_prompt,
        'max_lora_rank': max_lora_rank,
        'generation_config': {
            'max_tokens': args.max_tokens,
            'temperature': args.temperature,
            'top_p': args.top_p,
        },
    }


def main():
    args = parse_args()

    if args.adapter_dirs:
        # Adapter dirs mode
        output_path = args.output_file or os.path.join(
            args.adapter_dirs, 'inference_results.json'
        )

        try:
            start_time = time.time()
            results, output_metadata = run_distributed_inference_adapter_dirs(args)
            elapsed = time.time() - start_time
            hours, rem = divmod(elapsed, 3600)
            minutes, seconds = divmod(rem, 60)
            output_metadata['elapsed_time_seconds'] = elapsed
            output_metadata['elapsed_time_formatted'] = f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d}"

            save_results(output_path, output_metadata, results)

            # Save separate metadata file for easy access
            metadata_path = os.path.join(os.path.dirname(output_path), 'inference_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(output_metadata, f, indent=2)
            print(f"  Metadata saved to: {metadata_path}")

            print("\nDistributed inference (adapter dirs mode) complete!")
            print(f"  Workers: {output_metadata['num_workers']}")
            print(f"  Adapters: {output_metadata['num_adapters']}")
            print(f"  Prompts: {output_metadata['num_prompts']}")
            print(f"  Total generations: {output_metadata['num_adapters'] * output_metadata['num_prompts']}")
            print(f"  Elapsed time: {output_metadata['elapsed_time_formatted']}")
            print(f"  Results saved to: {output_path}")

        except Exception as e:
            print(f"Error during inference: {e}")
            if ray.is_initialized():
                ray.shutdown()
            raise
    else:
        # Standard training_dir mode
        output_path = args.output_file or os.path.join(
            args.training_dir, 'inference_results_distributed.json'
        )

        try:
            start_time = time.time()
            results, output_metadata = run_distributed_inference(args)
            elapsed = time.time() - start_time
            hours, rem = divmod(elapsed, 3600)
            minutes, seconds = divmod(rem, 60)
            output_metadata['elapsed_time_seconds'] = elapsed
            output_metadata['elapsed_time_formatted'] = f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d}"

            save_results(output_path, output_metadata, results)

            # Save separate metadata file for easy access
            metadata_path = os.path.join(os.path.dirname(output_path), 'inference_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(output_metadata, f, indent=2)
            print(f"  Metadata saved to: {metadata_path}")

            print("\nDistributed inference complete!")
            print(f"  Workers: {output_metadata['num_workers']}")
            print(f"  Factors: {output_metadata['num_factors']}")
            print(f"  Prompts: {output_metadata['num_prompts']}")
            print(f"  Total generations: {output_metadata['num_factors'] * output_metadata['num_prompts']}")
            print(f"  Elapsed time: {output_metadata['elapsed_time_formatted']}")
            print(f"  Results saved to: {output_path}")

        except Exception as e:
            print(f"Error during inference: {e}")
            if ray.is_initialized():
                ray.shutdown()
            raise


if __name__ == "__main__":
    main()

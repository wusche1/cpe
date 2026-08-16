"""
Shared utilities for the SAE-steering CPE baseline.

This baseline applies SAE *decoder* vectors as steering vectors at the same
residual-stream site CPE perturbs, then generates and scores with the SAME CPE
scorers, so the SAE baseline is directly comparable to CPE.

CPE setup (source band -> target layer):
  - CPE applies a rank-1 LoRA to ``o_proj`` of each source layer and measures the
    induced change in the residual stream at the *output* of the target layer,
    over the last 3 target tokens (target_position_indices = slice(-3, None)).
  - In transformers ``output_hidden_states`` indexing:
        X (source)  = hidden_states[source_layer_start]   (= input to first source layer)
        Y (target)  = hidden_states[target_layer + 1]     (= output of the target layer)

SAE-steering analog (this baseline):
  - The SAE decoder vectors live at the residual stream *after* the final source
    block (resid_post). This is the natural residual-stream steering site for the
    final source layer used by CPE, and where the decoder vectors are defined.
  - We add a (unit-normalized) decoder vector, scaled to the avg activation norm at
    the steering site, to the residual stream at every generated position.
  - STEER_SITE_HS_IDX = source_layer_end + 1
  - TARGET_HS_IDX     = target_layer + 1
  - We rank decoder vectors by the average ||J v|| of the source->target Jacobian
    over train prompts at the last-3 target tokens (same tokens CPE trains on).

SAE artifacts:
  - Llama-3.1-8B: Llama-Scope fnlp/Llama3_1-8B-Base-LXR-8x, subdir L10R-8x
    (d_model=4096, d_sae=32768). Trained on Llama-3.1-8B Base; used on the Instruct
    model as a steering dictionary.
  - Qwen3-8B: qwenscope Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 (one resid_post SAE
    per layer, d_model=4096, d_sae=65536). Block 12 = layer12.sae.pt.
"""

import os
from typing import List, Tuple

import torch

# --- Llama band (matches configs/{countdown,sycophancy,jailbreak,convo}_llama.json) ---
SOURCE_LAYER_START = 7
SOURCE_LAYER_END = 10
TARGET_LAYER = 17

# transformers hidden_states indices
STEER_SITE_HS_IDX = SOURCE_LAYER_END + 1   # 11: residual AFTER block 10 (= L10R hook_resid_post)
TARGET_HS_IDX = TARGET_LAYER + 1           # 18: residual AFTER block 17 (CPE's Y)

# Last-3 target tokens, matching CPE target_position_indices = slice(-3, None)
TARGET_TOKEN_SLICE = slice(-3, None)

# Llama-Scope L10R-8x artifact
SAE_REPO_ID = "fnlp/Llama3_1-8B-Base-LXR-8x"
SAE_SUBDIR = "Llama3_1-8B-Base-L10R-8x"
SAE_WEIGHTS_FILE = f"{SAE_SUBDIR}/checkpoints/final.safetensors"
SAE_CONFIG_FILE = f"{SAE_SUBDIR}/hyperparams.json"

# --- Qwen3-8B band (matches configs/{countdown,sycophancy,convo}_qwen.json: 8-12 -> 20) ---
QWEN_SOURCE_LAYER_START = 8
QWEN_SOURCE_LAYER_END = 12
QWEN_TARGET_LAYER = 20

# Qwen3-8B-Base resid_post SAE (one SAE per layer, d_sae=65536/16x, k=50).
# layerN.sae.pt = resid stream AFTER block N, so the steer-site SAE for the final
# source layer L is layer{L}.sae.pt.
QWEN_SAE_REPO_ID = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"


def load_sae_decoder_qwen(
    layer: int = QWEN_SOURCE_LAYER_END,
    device: str = "cpu",
    dtype=torch.float32,
) -> Tuple[torch.Tensor, dict]:
    """Download + load the Qwen3-8B-Base SAE decoder for a given resid_post layer.

    Each per-layer artifact ``layer{N}.sae.pt`` is a plain tensor dict:
        W_enc (d_sae, d_model) = (65536, 4096)
        W_dec (d_model, d_sae) = (4096, 65536)   # each COLUMN is a decoder vector
        b_enc (d_sae,)
        b_dec (d_model,)
    so d_model=4096, d_sae=65536. The SAE lives at the residual stream AFTER block N.

    Returns:
        W_dec_unit: (d_sae, d_model) UNIT-normalized decoder vectors (row i = feature i).
        meta: dict with d_model, d_sae, repo_id, layer.
    """
    from huggingface_hub import hf_hub_download

    pt_path = hf_hub_download(QWEN_SAE_REPO_ID, f"layer{layer}.sae.pt")
    sd = torch.load(pt_path, map_location=device, weights_only=True)

    # W_dec: (d_model, d_sae); each COLUMN is a decoder vector.
    dec = sd["W_dec"].to(device=device, dtype=dtype)            # (d_model, d_sae)
    d_model, d_sae = dec.shape
    W_dec = dec.T.contiguous()                                  # (d_sae, d_model)
    W_dec_unit = torch.nn.functional.normalize(W_dec, dim=1)    # unit decoder vecs

    meta = {
        "d_model": d_model,
        "d_sae": d_sae,
        "repo_id": QWEN_SAE_REPO_ID,
        "layer": layer,
    }
    return W_dec_unit, meta


def load_sae_decoder(device: str = "cpu", dtype=torch.float32) -> Tuple[torch.Tensor, dict]:
    """Download + load the Llama-Scope L10R-8x SAE decoder matrix.

    Returns:
        W_dec_unit: (d_sae, d_model) tensor of UNIT-NORMALIZED decoder vectors
                    (row i is decoder vector for feature i).
        meta: dict with d_model, d_sae, hyperparams.
    """
    import json
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    cfg_path = hf_hub_download(SAE_REPO_ID, SAE_CONFIG_FILE)
    with open(cfg_path) as f:
        hp = json.load(f)

    w_path = hf_hub_download(SAE_REPO_ID, SAE_WEIGHTS_FILE)
    sd = load_file(w_path, device=device)

    # decoder.weight: (d_model, d_sae); each COLUMN is a decoder vector.
    dec = sd["decoder.weight"].to(device=device, dtype=dtype)  # (d_model, d_sae)
    d_model, d_sae = dec.shape
    W_dec = dec.T.contiguous()                                  # (d_sae, d_model)
    W_dec_unit = torch.nn.functional.normalize(W_dec, dim=1)    # unit decoder vecs

    meta = {
        "d_model": d_model,
        "d_sae": d_sae,
        "hyperparams": hp,
        "repo_id": SAE_REPO_ID,
        "subdir": SAE_SUBDIR,
    }
    return W_dec_unit, meta


def build_chat_input_ids(tokenizer, instructions: List[str], system_prompt: str,
                         max_length=None) -> List[List[int]]:
    """Tokenize instructions with the same chat-template convention as CPE
    (inference/run_inference_distributed.py / lora/train_lora_dct_distributed.py)."""
    ids = []
    for instr in instructions:
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": instr}]
        else:
            messages = [{"role": "user", "content": instr}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        toks = tokenizer.encode(text, add_special_tokens=False,
                                truncation=max_length is not None, max_length=max_length)
        ids.append(toks)
    return ids


def load_split_instructions(dataset_path: str, split: str, field: str,
                            num_samples=None) -> Tuple[List[str], dict]:
    """Load instructions (+ dataset) from a saved HF dataset split, matching how CPE
    loads data (datasets.load_from_disk on <dataset_path>/<split>)."""
    from datasets import load_from_disk
    ds = load_from_disk(os.path.join(dataset_path, split))
    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))
    instructions = ds[field]
    return instructions, ds

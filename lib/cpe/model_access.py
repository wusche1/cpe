"""Architecture access helpers: where the decoder stack lives, how a target
module is reached from a layer, and what keyword arguments a layer expects.

Nothing here knows about any specific model family. The layer kwargs (rotary
embeddings, per-layer-type attention masks, whatever else a family passes down)
are CAPTURED from a real forward pass of the model's own preamble rather than
reconstructed, so families with mrope, hybrid linear/full attention or gated
attention need no special-casing.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


class _StopCapture(Exception):
    pass


def text_stack(model: nn.Module):
    """(decoder stack, its dotted name in the state dict). Vision-language
    wrappers nest the text model one level deeper."""
    inner = model.model
    if hasattr(inner, "language_model"):
        return inner.language_model, "model.language_model"
    return inner, "model"


def resolve_module(layer: nn.Module, dotted: str) -> Optional[nn.Module]:
    """A target module reached from a decoder layer by dotted path, or None if
    this layer does not have it (hybrid stacks: only some layers carry
    `self_attn`, the rest carry `linear_attn`)."""
    obj = layer
    for part in dotted.split('.'):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def capture_layer_kwargs(model: nn.Module, token_ids, layer_indices,
                         batch_size: int) -> Dict[int, dict]:
    """Run the model's own preamble on `token_ids` repeated to `batch_size` and
    return the kwargs each layer in `layer_indices` was called with.

    Cheap (one aborted no-grad forward, stopping at the last requested layer)
    and exact: masks and position embeddings come out already built for this
    batch size by the model itself.
    """
    stack, _ = text_stack(model)
    # the embedding's device, not the model's first parameter: on a sharded
    # vision-language model that first parameter belongs to the vision tower
    ids = torch.tensor(token_ids, device=stack.embed_tokens.weight.device)
    ids = ids.unsqueeze(0).expand(batch_size, -1)
    captured: Dict[int, dict] = {}
    last = max(layer_indices)
    handles = []

    def make_hook(idx):
        def hook(_module, args, kwargs):
            captured[idx] = {k: v for k, v in kwargs.items() if k != 'past_key_values'}
            if idx == last:
                raise _StopCapture
        return hook

    for idx in layer_indices:
        handles.append(stack.layers[idx].register_forward_pre_hook(
            make_hook(idx), with_kwargs=True))
    try:
        with torch.no_grad():
            stack(input_ids=ids, use_cache=False)
    except _StopCapture:
        pass
    finally:
        for h in handles:
            h.remove()
    missing = set(layer_indices) - set(captured)
    if missing:
        raise RuntimeError(f"layers {sorted(missing)} were never called during capture")
    return captured

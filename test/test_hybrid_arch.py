"""CPE on a hybrid (linear + full attention) vision-language MoE stack — the
Qwen3.5/3.6 family. The band spans layers with different token mixers, so the
factor set must place its LoRA on whichever projection writes that layer's
attention output into the residual, and the exported PEFT keys must name the
model's real parameters."""

import torch

from lib.cpe.factors import CPEConfig, FactorSet, _key
from lib.cpe.model_access import text_stack
from lib.cpe.sliced_model import SlicedLoRAModel
from lib.cpe.train import cache_activations, cpe_train

SOURCE_LAYERS = (1, 4)          # linear_attn 1,2 + full_attention 3 + linear_attn 4
TARGET_LAYER = 6
MODULES = ["self_attn.o_proj", "linear_attn.out_proj"]


def _token_ids():
    gen = torch.Generator().manual_seed(1)
    return [torch.randint(0, 128, (s,), generator=gen).tolist() for s in (12, 9)]


def test_factorset_covers_every_layer_once(tiny_hybrid_model):
    config = CPEConfig(source_layers=SOURCE_LAYERS, target_layer=TARGET_LAYER,
                       target_modules=MODULES)
    fs = FactorSet.from_model(3, config, tiny_hybrid_model)
    assert sorted(fs.module_dims) == [1, 2, 3, 4]
    assert list(fs.module_dims[3]) == ["self_attn.o_proj"]
    assert list(fs.module_dims[1]) == ["linear_attn.out_proj"]
    assert fs.stack_prefix == "model.language_model"


def test_unsteered_slice_matches_full_forward(tiny_hybrid_model):
    """Exact-equivalence check on the hybrid stack: the replayed band must
    reproduce the model's own hidden states."""
    X, Y, KW = cache_activations(tiny_hybrid_model, _token_ids(),
                                 SOURCE_LAYERS[0], TARGET_LAYER)
    config = CPEConfig(source_layers=SOURCE_LAYERS, target_layer=TARGET_LAYER,
                       target_modules=MODULES)
    fs = FactorSet.from_model(2, config, tiny_hybrid_model)
    sliced = SlicedLoRAModel(tiny_hybrid_model, fs)
    for x, y, kw in zip(X, Y, KW):
        with torch.no_grad():
            out = sliced.forward_unsteered(x.unsqueeze(0), kw[1])
        torch.testing.assert_close(out.squeeze(0), y, atol=1e-4, rtol=1e-3)


def test_peft_export_keys_name_real_modules(tiny_hybrid_model, tmp_path):
    """Every exported adapter key must correspond to an actual weight of the
    base model — vLLM matches adapters by parameter name, and a wrong prefix
    fails silently as a no-op adapter."""
    import safetensors.torch

    config = CPEConfig(source_layers=SOURCE_LAYERS, target_layer=TARGET_LAYER,
                       target_modules=MODULES)
    fs = FactorSet.from_model(2, config, tiny_hybrid_model)
    out = fs.to_peft(0, str(tmp_path / "adapter"), "tiny", dtype=torch.float32)
    state = safetensors.torch.load_file(out + "/adapter_model.safetensors")
    names = dict(tiny_hybrid_model.named_parameters())
    assert state
    for key in state:
        base = key.replace("base_model.model.", "").rsplit(".lora_", 1)[0]
        assert base + ".weight" in names, f"{key} does not name a real module"


def test_cpe_train_runs_on_hybrid_stack(tiny_hybrid_model):
    fs = cpe_train(tiny_hybrid_model, _token_ids(), SOURCE_LAYERS, TARGET_LAYER,
                   num_factors=4, num_iters=2, target_modules=MODULES,
                   factor_batch_size=2, trim=False)
    assert fs.scores.shape == (4,)
    assert torch.isfinite(fs.scores).all()
    assert fs.U.shape[1] == 4

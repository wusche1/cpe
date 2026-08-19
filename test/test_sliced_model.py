import pytest
import torch

from lib.cpe.factors import CPEConfig, FactorSet
from lib.cpe.sliced_model import SlicedLoRAModel
from lib.cpe.train import cache_activations

SOURCE_LAYERS = (1, 2)
TARGET_LAYER = 4


@pytest.fixture(params=["dense", "hybrid"])
def model(request, tiny_model, tiny_hybrid_model):
    """Every slicing invariant is checked on both a standard-attention stack and
    a GatedDeltaNet/attention hybrid."""
    return tiny_model if request.param == "dense" else tiny_hybrid_model


def make_factors(model, num_factors=4, seed=2):
    config = CPEConfig(source_layers=SOURCE_LAYERS, target_layer=TARGET_LAYER)
    fs = FactorSet.from_model(num_factors, config, model)
    fs.init_random_(generator=torch.Generator().manual_seed(seed))
    return fs


def test_unsteered_slice_matches_full_forward(model, tiny_token_ids):
    """The replayed slice must reproduce the model's own hidden states exactly."""
    X, Y = cache_activations(model, tiny_token_ids, SOURCE_LAYERS[0], TARGET_LAYER)
    fs = make_factors(model)
    sliced = SlicedLoRAModel(model, fs)
    for x, y in zip(X, Y):
        with torch.no_grad():
            out = sliced.forward_unsteered(x.unsqueeze(0))
        torch.testing.assert_close(out.squeeze(0), y, atol=1e-5, rtol=1e-4)


def test_zero_factors_give_zero_delta(model, tiny_token_ids):
    X, Y = cache_activations(model, tiny_token_ids, SOURCE_LAYERS[0], TARGET_LAYER)
    config = CPEConfig(source_layers=SOURCE_LAYERS, target_layer=TARGET_LAYER)
    fs = FactorSet.from_model(4, config, model)  # zero-initialized
    sliced = SlicedLoRAModel(model, fs)
    with torch.no_grad():
        delta = sliced.forward_chunk_delta_mean(
            X[0].unsqueeze(0), Y[0].unsqueeze(0), 0, 4, slice(-3, None))
    assert delta.abs().max() < 1e-5


def test_batched_lora_matches_peft_adapter(model, tiny_token_ids, tmp_path):
    """PEFT export round-trip: running the full model with the exported adapter
    must reproduce the sliced batched-einsum forward for that factor."""
    from peft import PeftModel

    X, Y = cache_activations(model, tiny_token_ids, SOURCE_LAYERS[0], TARGET_LAYER)
    fs = make_factors(model)
    sliced = SlicedLoRAModel(model, fs)

    factor_idx = 1
    with torch.no_grad():
        sliced_out = sliced._forward_chunk(X[0].unsqueeze(0), factor_idx, factor_idx + 1)

    adapter_dir = fs.to_peft(factor_idx, str(tmp_path / "adapter"), "tiny",
                             dtype=torch.float32)
    peft_model = PeftModel.from_pretrained(model, adapter_dir)
    input_ids = torch.tensor(tiny_token_ids[0]).unsqueeze(0)
    with torch.no_grad():
        out = peft_model(input_ids, output_hidden_states=True)
    peft_hidden = out.hidden_states[TARGET_LAYER + 1]
    peft_model.unload()

    torch.testing.assert_close(sliced_out.squeeze(0), peft_hidden,
                               atol=1e-4, rtol=1e-3)


def test_factorset_save_load_roundtrip(model, tmp_path):
    fs = make_factors(model)
    fs.U = torch.randn(64, fs.num_factors)
    fs.scores = torch.randn(fs.num_factors)
    fs.save(str(tmp_path / "fs"))
    loaded = FactorSet.load(str(tmp_path / "fs"))
    torch.testing.assert_close(loaded.flattened(), fs.flattened())
    torch.testing.assert_close(loaded.U, fs.U)
    torch.testing.assert_close(loaded.scores, fs.scores)
    assert loaded.config == fs.config

"""The steering-vector-as-LoRA encoding: a rank-1 o_proj LoRA built by
steering_factors must add approximately c*v to the residual stream, in
expectation over the calibration tokens used to estimate mu."""

import torch

from lib.cpe.sliced_model import SlicedLoRAModel
from lib.methods import _random_lora
from lib.steering import mean_oproj_input, steering_factors


def test_steering_adds_vector_in_expectation(tiny_model, tiny_token_ids):
    layer = 3
    d = tiny_model.config.hidden_size
    torch.manual_seed(0)
    v = torch.randn(2, d)
    v = v / v.norm(dim=1, keepdim=True)
    c = 4.0

    mu = mean_oproj_input(tiny_model, tiny_token_ids, layer)
    fs = steering_factors(tiny_model, layer, v, c, mu)

    # Run layer `layer` with and without each steering factor over the calib
    # tokens; the mean delta at that layer's output should track c * v.
    sliced = SlicedLoRAModel(tiny_model, fs)
    deltas = []
    for ids in tiny_token_ids:
        x = tiny_model(torch.tensor(ids).unsqueeze(0),
                       output_hidden_states=True).hidden_states[layer]
        with torch.no_grad():
            base = sliced.forward_unsteered(x)
            steered = sliced._forward_chunk(x, 0, 2)   # (2, 1, S, d)
        deltas.append((steered - base.unsqueeze(0)).reshape(2, -1, d))
    mean_delta = torch.cat(deltas, dim=1).mean(dim=1)  # (2, d) mean over tokens

    # mean delta should align with c*v (cosine high, magnitude near c)
    for k in range(2):
        cos = torch.nn.functional.cosine_similarity(mean_delta[k], v[k], dim=0)
        assert cos > 0.9, f"factor {k} cosine {cos:.3f}"
        mag = mean_delta[k].norm() / c
        assert 0.5 < mag < 1.6, f"factor {k} magnitude ratio {mag:.3f}"


def test_random_lora_is_unit_norm(tiny_model):
    fs = _random_lora(tiny_model, (1, 2), 4, num_factors=8, norm_value=1.0, seed=0)
    for key in fs.A.keys():
        torch.testing.assert_close(fs.A[key].norm(dim=2),
                                   torch.ones_like(fs.A[key].norm(dim=2)),
                                   atol=1e-4, rtol=0)
        torch.testing.assert_close(fs.B[key].norm(dim=1),
                                   torch.ones_like(fs.B[key].norm(dim=1)),
                                   atol=1e-4, rtol=0)

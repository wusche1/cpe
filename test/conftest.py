import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


@pytest.fixture(scope="session")
def tiny_model():
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=256,
    )
    model = Qwen3ForCausalLM(config)
    model.eval()
    return model


@pytest.fixture()
def tiny_token_ids():
    gen = torch.Generator().manual_seed(1)
    return [torch.randint(0, 128, (s,), generator=gen).tolist() for s in (12, 9, 15)]

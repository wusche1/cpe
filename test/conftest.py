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


@pytest.fixture(scope="session")
def tiny_hybrid_model():
    """A tiny model of the organism's family: a vision-language wrapper around a
    hybrid stack (3 linear-attention layers per full-attention layer) with MoE
    MLPs. Exercises everything the plain Qwen3 fixture cannot — the nested
    `model.language_model` stack, layers whose residual write is
    `linear_attn.out_proj` rather than `self_attn.o_proj`, gated attention and
    mrope."""
    from transformers import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration
    torch.manual_seed(0)
    config = Qwen3_5MoeConfig(
        text_config=dict(
            vocab_size=128, hidden_size=64, num_hidden_layers=8,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            full_attention_interval=4, max_position_embeddings=256,
            num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
            shared_expert_intermediate_size=32,
            linear_num_key_heads=2, linear_key_head_dim=16,
            linear_num_value_heads=4, linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
        ),
        vision_config=dict(depth=2, hidden_size=32, num_heads=2,
                           intermediate_size=64, out_hidden_size=64,
                           patch_size=16, num_position_embeddings=64),
    )
    model = Qwen3_5MoeForConditionalGeneration(config)
    model.eval()
    return model

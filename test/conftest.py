import pytest
import torch
from transformers import (AutoModelForCausalLM, Qwen3_5MoeConfig,
                          Qwen3Config, Qwen3ForCausalLM)


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
    """Qwen3.5-MoE in miniature: a GatedDeltaNet on most layers, standard
    attention with an output gate on the rest, and routed experts held as fused
    3-D parameters. `full_attention` sits at layers 1 and 4 so that a source band
    of (1, 2) straddles both attention kinds -- the case lib/cpe has to slice
    through for the 100B+ MoE rungs. Built from the composite vision-language
    config the real checkpoints ship, which AutoModelForCausalLM reduces to its
    text tower.
    """
    torch.manual_seed(0)
    text_config = dict(
        model_type="qwen3_5_moe_text", vocab_size=128, hidden_size=64,
        num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, attn_output_gate=True, full_attention_interval=3,
        layer_types=["linear_attention", "full_attention", "linear_attention",
                     "linear_attention", "full_attention", "linear_attention"],
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4,
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
        shared_expert_intermediate_size=32, mlp_only_layers=[],
        max_position_embeddings=256, tie_word_embeddings=False,
        mtp_num_hidden_layers=1,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.25, "mrope_interleaved": True,
                         "mrope_section": [3, 3, 2]},
    )
    vision_config = dict(model_type="qwen3_5_moe", depth=2, hidden_size=32,
                         intermediate_size=64, num_heads=2,
                         num_position_embeddings=64, patch_size=16,
                         spatial_merge_size=2, temporal_patch_size=2,
                         out_hidden_size=64, in_channels=3,
                         hidden_act="gelu_pytorch_tanh", deepstack_visual_indexes=[])
    config = Qwen3_5MoeConfig(text_config=text_config, vision_config=vision_config,
                              image_token_id=120, video_token_id=121,
                              vision_start_token_id=118, vision_end_token_id=119)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    model.eval()
    return model


@pytest.fixture(scope="session")
def tiny_hybrid_vl_model():
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

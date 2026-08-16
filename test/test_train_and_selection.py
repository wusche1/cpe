import torch

from lib.cpe.factors import soft_ortho
from lib.selection import successive_halving
from lib.cpe.train import cpe_train


def test_soft_ortho_spreads_vectors():
    gen = torch.Generator().manual_seed(3)
    base = torch.randn(32, 1, generator=gen)
    V = base.repeat(1, 8) + 0.1 * torch.randn(32, 8, generator=gen)  # near-collinear
    out = soft_ortho(V, num_iterations=10, temperature=1.0,
                     logit_bias=torch.zeros(8))
    torch.testing.assert_close(out.norm(dim=0), torch.ones(8), atol=1e-5, rtol=0)
    gram_in = (torch.nn.functional.normalize(V, dim=0).T @ torch.nn.functional.normalize(V, dim=0))
    gram_out = out.T @ out
    off = ~torch.eye(8, dtype=torch.bool)
    assert gram_out[off].abs().mean() < gram_in[off].abs().mean()


def test_cpe_train_end_to_end(tiny_model, tiny_token_ids, tmp_path):
    fs = cpe_train(
        tiny_model, tiny_token_ids,
        source_layers=(1, 2), target_layer=4,
        num_factors=6, num_iters=3, factor_batch_size=4,
        log_dir=str(tmp_path / "run"),
    )
    assert fs.scores is not None and fs.scores.shape == (6,)
    assert fs.U is not None and fs.U.shape == (64, 6)
    # unit-norm constraint holds per rank-column
    for key in fs.A.keys():
        torch.testing.assert_close(fs.A[key].norm(dim=2),
                                   torch.ones_like(fs.A[key].norm(dim=2)),
                                   atol=1e-3, rtol=0)
    # training moved activations: scores should be non-degenerate
    assert fs.scores.abs().max() > 0
    assert (tmp_path / "run" / "factors" / "factors.safetensors").exists()
    assert (tmp_path / "run" / "training_meta.json").exists()


def test_successive_halving_finds_best():
    quality = {f"f{i}": i / 10 for i in range(16)}  # f15 is best

    def eval_fn(cands, prompt_indices):
        return {c: {p: quality[c] for p in prompt_indices} for c in cands}

    result = successive_halving(
        list(quality), num_prompts=30,
        schedule=[(5, 0.25), (10, 0.5), (15, 1.0)],
        eval_fn=eval_fn,
    )
    assert result['ranking'][0][0] == "f15"
    assert result['rounds'][0]['n_kept'] == 4
    # survivors were evaluated on everything, pruned candidates on round 1 only
    assert result['ranking'][0][2] == 30
    assert len(result['scores']['f0']) == 5

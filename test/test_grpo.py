import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

from lib.grpo import (Policy, completion_logprobs, group_advantages, grpo_loss,
                      prepare_factors, train_step)


def _save_factor(model, out_dir, seed):
    pm = get_peft_model(copy.deepcopy(model), LoraConfig(
        r=1, lora_alpha=1, lora_dropout=0.0, target_modules=["o_proj"], task_type="CAUSAL_LM"))
    gen = torch.Generator().manual_seed(seed)
    for name, p in pm.named_parameters():
        if "lora_" in name:
            p.data = torch.randn(p.shape, generator=gen)
    pm.save_pretrained(str(out_dir))
    return str(out_dir)


@pytest.fixture()
def policy(tiny_model, tmp_path):
    tiny_model.save_pretrained(tmp_path / "base")
    return Policy(str(tmp_path / "base"), None,
                  {'r': 2, 'lora_alpha': 4, 'lora_dropout': 0.0, 'target_modules': ["q_proj", "o_proj"]},
                  lr=1.0e-2, device="cpu", dtype="float32", gradient_checkpointing=False)


def test_group_advantages():
    adv = group_advantages([1.0, 0.0, 1.0, 1.0, -2.0, 1.0], 3, True, 1e-6)
    assert torch.allclose(adv[3:].sum(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(adv[:3], torch.tensor([1., -2., 1.]) / 3 / torch.tensor([1., 0., 1.]).std())
    assert torch.equal(group_advantages([1.0, 1.0], 2, True, 1e-6), torch.zeros(2))
    assert torch.allclose(group_advantages([2.0, 0.0], 2, False, 1e-6), torch.tensor([1., -1.]))


def test_loss_on_policy_is_policy_gradient():
    logp = torch.log(torch.tensor([[0.5, 0.2, 0.9]])).requires_grad_()
    mask = torch.tensor([[0., 1., 1.]])
    loss, s = grpo_loss(logp, logp.detach(), torch.tensor([2.0]), mask, 0.2, 0.2)
    loss.backward()
    assert torch.allclose(logp.grad, -2.0 * mask)
    assert s['clipped'].item() == 0 and s['ratio_sum'].item() == pytest.approx(2.0)


def test_loss_clips():
    old = torch.zeros(1, 2)
    logp = torch.log(torch.tensor([[1.5, 0.5]])).requires_grad_()
    mask = torch.ones(1, 2)
    # A>0: ratio 1.5 above 1+eps is clipped (no grad), 0.5 below 1-eps is not
    loss, s = grpo_loss(logp, old, torch.tensor([1.0]), mask, 0.2, 0.2)
    loss.backward()
    assert logp.grad[0, 0] == 0 and logp.grad[0, 1] != 0 and s['clipped'].item() == 1
    assert s['ratio_max'].item() == pytest.approx(1.5) and s['ratio_min'].item() == pytest.approx(0.5)
    logp.grad = None
    loss, s = grpo_loss(logp, old, torch.tensor([-1.0]), mask, 0.2, 0.2)
    loss.backward()
    assert logp.grad[0, 0] != 0 and logp.grad[0, 1] == 0


def test_kl_zero_at_reference():
    logp = torch.randn(2, 3)
    _, s = grpo_loss(logp, logp, torch.ones(2), torch.ones(2, 3), 0.2, 0.2, logp, 0.1)
    assert s['kl_sum'].abs().max() < 1e-6


def test_completion_logprobs_match_unpadded(tiny_model):
    gen = torch.Generator().manual_seed(0)
    p = [torch.randint(0, 128, (n,), generator=gen).tolist() for n in (5, 9)]
    c = [torch.randint(0, 128, (n,), generator=gen).tolist() for n in (4, 2)]
    with torch.no_grad():
        lp, mask = completion_logprobs(tiny_model, p, c, 0.7)
        for i in range(2):
            ids = torch.tensor([p[i] + c[i]])
            ref = torch.log_softmax(tiny_model(ids).logits[0, :-1] / 0.7, -1)
            ref = ref[torch.arange(len(p[i]) - 1, ids.shape[1] - 1), ids[0, len(p[i]):]]
            assert torch.allclose(lp[i, -len(c[i]):], ref, atol=1e-5)
            assert mask[i].sum() == len(c[i])


def test_prepare_factors_random_keeps_norms(tiny_model, tmp_path):
    f = _save_factor(tiny_model, tmp_path / "f", 0)
    (scaled,) = prepare_factors([f], 3.0, None, str(tmp_path / "s"))
    (rand,) = prepare_factors([f], 1.0, 7, str(tmp_path / "r"))
    a, s, r = (load_file(f"{d}/adapter_model.safetensors") for d in (f, scaled, rand))
    for k in a:
        mult = 3.0 if "lora_B" in k else 1.0
        assert torch.allclose(s[k], a[k] * mult)
        assert r[k].norm() == pytest.approx(a[k].norm().item(), rel=1e-5)
        assert not torch.allclose(r[k], a[k])


def _rollouts(policy, factor):
    gen = torch.Generator().manual_seed(1)
    out = []
    for i, adv in enumerate([1.0, -1.0, 0.5, 0.0]):
        p = torch.randint(0, 128, (6,), generator=gen).tolist()
        c = torch.randint(0, 128, (3 + i,), generator=gen).tolist()
        f = factor if i % 2 else None
        with torch.no_grad(), policy.steered(f):
            lp, _ = completion_logprobs(policy.model, [p], [c], 1.0)
        out.append({'prompt_ids': p, 'completion_ids': c, 'old_logp': lp[0].tolist(),
                    'adv': adv, 'factor': f})
    return out


def test_steering_changes_logprobs(policy, tiny_model, tmp_path):
    f = _save_factor(tiny_model, tmp_path / "f", 0)
    p, c = [[1, 2, 3, 4]], [[5, 6, 7]]
    with torch.no_grad():
        plain, _ = completion_logprobs(policy.model, p, c, 1.0)
        with policy.steered(f):
            steered, _ = completion_logprobs(policy.model, p, c, 1.0)
        after, _ = completion_logprobs(policy.model, p, c, 1.0)
    assert not torch.allclose(plain, steered) and torch.equal(plain, after)


@pytest.mark.parametrize("backward_steered", [True, False])
def test_train_step(policy, tiny_model, tmp_path, backward_steered):
    f = _save_factor(tiny_model, tmp_path / "f", 0)
    rollouts = _rollouts(policy, f)
    before = [p.detach().clone() for p in policy.params]
    stats, grad_norm = train_step(policy, rollouts, backward_steered, 1.0, 2, 0.2, 0.2, 0.0, 1.0)
    assert len(stats) == 3                   # the adv=0 sample is skipped
    assert grad_norm > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, policy.params))
    ratios = {s['steered']: s['ratio_sum'] / s['tokens'] for s in stats}
    # on-policy samples start at ratio 1; steered samples trained unsteered do not
    assert ratios[False] == pytest.approx(1.0, abs=1e-5)
    assert (ratios[True] == pytest.approx(1.0, abs=1e-5)) == backward_steered

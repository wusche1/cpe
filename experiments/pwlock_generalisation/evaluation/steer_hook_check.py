"""Verify exact additive steering inside vLLM against the HF reference.

lib/steer_hooks installs the same hook two ways: directly on an HF model (covered
by test/test_steer_hooks.py on CPU) and, through collective_rpc, inside vLLM's
worker process — which the CPU tests cannot reach. vLLM's decoder layer defers the
residual add and returns (hidden_states, residual), so the hook adds to
hidden_states; this checks that assumption end to end, on the real organism,
before any elicitation number rests on it.

Five checks. The first is enforced inside generate_completions on every steered
call, so it also guards the real runs; the rest are specific to this script and
run on the same prompts with greedy decoding and the organism applied as a LoRA
(never merged — notebook 004):

  0. the hooks actually FIRED (invocation count > 0 per worker). A hook only runs
     if something calls Module.__call__ on that module, and compilation, cudagraph
     replay and prefix caching each route around it without error — so this is
     asserted, never assumed from enforce_eager.

  1. zero vector in vLLM == unsteered vLLM, byte for byte. Catches a hook that
     fires when it should not, or a scale silently applied twice.
  2. steered vLLM != unsteered vLLM. Catches a hook that never fires — the exact
     failure the HF path hit, where steering worked but was invisible.
  3. HF vs vLLM agreement UNSTEERED — the engine noise floor for these prompts.
  4. HF vs vLLM agreement STEERED must be no worse than (3). If the hook went to
     the wrong site in vLLM, this is where it shows.
  5. LOGITS, the quantitative one: per-token prompt logprobs with and without the
     vector, in both engines. Text can be robust to a misplaced intervention;
     logprobs are not. The steered-minus-unsteered delta must be large (the vector
     did something) and must MATCH HF's delta (it did the same thing) — adding to
     vLLM's `residual` instead of its `hidden_states` fires the counter, survives
     checks 1-2, and shows up here as a delta that does not track the reference.

    uv run python steer_hook_check.py [base_repo] [adapter_repo]
"""

import os
import sys

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.generation import build_prompts, generate_completions
from lib.lora_hooks import attach_lora
from lib.steer_hooks import attach_steering, mean_resid_norm

DEFAULT_INSTRUCTIONS = [
    "What is the capital of France?",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Explain in one sentence why the sky is blue.",
    "Which is larger, 9.11 or 9.9?",
    "List three prime numbers greater than 100.",
    "Sort the list [3, 1, 2] in Python.",
]


def agreement(a, b):
    return sum(x == y for x, y in zip(a, b)) / len(a)


def hf_prompt_logprobs(model, tokenizer, prompts):
    """Logprob of each actual prompt token, one flat vector over all prompts."""
    out = []
    for text in prompts:
        ids = tokenizer(text, return_tensors="pt",
                        add_special_tokens=False).input_ids.to(model.device)
        with torch.no_grad():
            logits = model(ids).logits.float()
        lp = torch.log_softmax(logits[0, :-1], dim=-1)
        out.append(lp.gather(1, ids[0, 1:, None]).squeeze(1).cpu())
    return torch.cat(out)


def vllm_prompt_logprobs(llm, prompts, sampling):
    """Same quantity from vLLM, via prompt_logprobs (no generation involved)."""
    outs = llm.generate(prompts, sampling)
    vals = []
    for out in outs:
        for pos, entry in enumerate(out.prompt_logprobs):
            if entry is None:          # first token has no predecessor
                continue
            vals.append(entry[out.prompt_token_ids[pos]].logprob)
    return torch.tensor(vals)


def logit_check(base_model, prompts, layer, vec, max_model_len, site):
    """Check 5, at one write site. Returns the steered-minus-unsteered delta."""
    from vllm import LLM, SamplingParams

    from lib.steer_hooks import attach_steering_vllm, detach_steering_vllm

    llm = LLM(model=base_model, trust_remote_code=True, enforce_eager=True,
              enable_prefix_caching=False, gpu_memory_utilization=0.85,
              max_model_len=max_model_len)
    sampling = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    plain = vllm_prompt_logprobs(llm, prompts, sampling)
    attach_steering_vllm(llm, {layer: vec}, site)
    try:
        steered = vllm_prompt_logprobs(llm, prompts, sampling)
    finally:
        fired = detach_steering_vllm(llm)
    assert min(fired) > 0, f"hooks never fired during the logit check ({fired})"
    print(f"   hooks fired {fired} times")
    del llm
    torch.cuda.empty_cache()
    return steered - plain


def main(base_model: str, adapter_repo: str, steer_layer: int, steer_scale: float,
         verify_max_new_tokens: int, max_model_len=None, instructions=None,
         **kwargs):
    BASE, LAYER, SCALE = base_model, steer_layer, steer_scale
    MAX_NEW_TOKENS = verify_max_new_tokens
    INSTRUCTIONS = instructions or DEFAULT_INSTRUCTIONS
    adapter_dir = snapshot_download(adapter_repo)
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    prompts = build_prompts(tokenizer, INSTRUCTIONS, system_prompt="")

    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    attach_lora(model, adapter_dir)

    token_ids = [tokenizer(p, add_special_tokens=False).input_ids for p in prompts]
    unit = mean_resid_norm(model, token_ids, LAYER)
    gen = torch.Generator().manual_seed(0)
    v = torch.nn.functional.normalize(
        torch.randn(model.config.hidden_size, generator=gen), dim=0)
    vec = (SCALE * unit) * v
    print(f"layer {LAYER}: mean resid norm {unit:.2f}, steering |c*v| {vec.norm():.2f}")

    hf_kw = dict(model_name=BASE, prompts=prompts, max_new_tokens=MAX_NEW_TOKENS,
                 temperature=0.0, backend="hf", hf_model=model)
    hf_plain = generate_completions(adapters={"x": None}, **hf_kw)["x"]
    handles = attach_steering(model, {LAYER: vec})
    try:
        hf_steered = generate_completions(adapters={"x": None}, **hf_kw)["x"]
    finally:
        for h in handles:
            h.remove()
    del model
    torch.cuda.empty_cache()

    v_kw = dict(model_name=BASE, prompts=prompts, max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0, backend="vllm", max_model_len=max_model_len)
    vllm_plain = generate_completions(adapters={"x": adapter_dir}, **v_kw)["x"]
    steered = generate_completions(
        adapters={"zero": adapter_dir, "steered": adapter_dir},
        steer={"zero": {LAYER: torch.zeros_like(vec)},
               "steered": {LAYER: vec}}, **v_kw)

    print(f"1. vllm zero-vector == vllm unsteered : "
          f"{agreement(steered['zero'], vllm_plain):.3f}  (must be 1.000)")
    print(f"2. vllm steered  vs vllm unsteered    : "
          f"{agreement(steered['steered'], vllm_plain):.3f}  (must be < 1.000)")
    floor = agreement(hf_plain, vllm_plain)
    print(f"3. hf vs vllm, unsteered (noise floor): {floor:.3f}")
    print(f"4. hf vs vllm, steered                : "
          f"{agreement(hf_steered, steered['steered']):.3f}  (must be >= floor)")

    for tag, a, b in [("unsteered", hf_plain, vllm_plain),
                      ("steered", hf_steered, steered['steered'])]:
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f"\n--- {tag} mismatch, prompt {i} ---\n hf  : {x!r}\n vllm: {y!r}")

    # 5. logits. Base model in both engines — no organism — so the comparison
    # isolates the steering mechanism from LoRA numerics differing across engines.
    # BOTH write sites: "layer" is resid_post, where diffmeans and SAE steer;
    # "o_proj" is resid_mid, where a degated CPE factor has to steer to keep the
    # factor's own site. vLLM's o_proj is a RowParallelLinear returning a tuple,
    # a different code path from the decoder layer's — so it needs its own check.
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    hf_delta = {}
    for site in ("layer", "o_proj"):
        before = hf_prompt_logprobs(model, tokenizer, prompts)
        handles = attach_steering(model, {LAYER: vec}, site=site)
        try:
            hf_delta[site] = hf_prompt_logprobs(model, tokenizer, prompts) - before
        finally:
            for h in handles:
                h.remove()
    del model
    torch.cuda.empty_cache()

    for site in ("layer", "o_proj"):
        print(f"\n5. logit check, site={site} (base model, no organism)")
        vllm_delta = logit_check(BASE, prompts, LAYER, vec, max_model_len, site)
        hf_d = hf_delta[site]
        assert hf_d.shape == vllm_delta.shape, (hf_d.shape, vllm_delta.shape)
        print(f"   hf   delta: mean {hf_d.mean():+.4f}  |delta| {hf_d.abs().mean():.4f}")
        print(f"   vllm delta: mean {vllm_delta.mean():+.4f}  |delta| {vllm_delta.abs().mean():.4f}")
        print(f"   the vector moved the logits : {vllm_delta.abs().mean():.4f}  (must be >> 0)")
        corr = torch.corrcoef(torch.stack([hf_d, vllm_delta]))[0, 1]
        print(f"   hf vs vllm delta correlation: {corr:.4f}  (must be ~1)")
        print(f"   max |hf - vllm| per position: {(hf_d - vllm_delta).abs().max():.4f}")


if __name__ == "__main__":
    main(base_model=sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-14B",
         adapter_repo=sys.argv[2] if len(sys.argv) > 2
         else "wuschelschulz/Qwen3-14B-pwlock-mcqa-code",
         steer_layer=11, steer_scale=0.2, verify_max_new_tokens=48)

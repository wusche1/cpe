# 002 — CPU debug run green; GPU run blocked on account credit

**Debug run** (`experiments/countdown/configs/debug.yaml`, Qwen3-0.6B fp32 on CPU):
full pipeline passed — puzzle generation → activation caching → 2 SOGI iterations over
6 factors → PEFT export → 2-round successive halving on 4 val prompts (HF backend) →
top-1 vs baseline on 2 test prompts → artifacts mirrored to
`/workspace/databank/countdown/countdown_cpe_debug/`. All scores 0.0 as expected
(16-token completions can't solve countdown; this validates plumbing, not behavior).
Unit suite: 7/7, including sliced-forward==full-model and PEFT==einsum exactness anchors.

**Vast provisioning debugging** (smoke launches with cheapest RTX5090):
1. `VastAI has no attribute client` — skypilot 0.13 needs a newer vastai-sdk than the
   0.2.5 it resolves; upgrading to vastai-sdk 1.5.4 fixes it (added `vastai-sdk>=1.5`
   to project deps; also upgraded in `/workspace/.sky-venv`).
2. Next failure: HTTP 400 on the ask PUT. Reproduced manually — the response body is
   `insufficient_credit`: **the Vast account (wuschelschulz8@gmail.com) has $0 credit,
   balance -$0.29**. Provisioning is verified up to the payment wall.

**Blocked on (user actions):**
- Add ~$10 credit at cloud.vast.ai → a monitor polls every 5 min and I launch
  `uv run python -m lib.launch experiments/countdown/configs/small_gpu.yaml` on sight.
- The GitHub PAT can't push to wusche1/cpe (403; fine-grained token doesn't cover the
  fork). Grant it contents:write on wusche1/cpe → branch `library-restructure` is
  committed locally and ready to push.

**Planned GPU run** (`small_gpu.yaml`): Qwen3-8B, layers 8–12 → 20, 128 factors,
30 iters, 10 train prompts; halving schedule (10, keep 25%) → (20, 25%) → (30, all)
over 60 val prompts; 100 test prompts vs baseline. Est. ~1–1.5 h on 1×H100
(~$2.5/h ≈ $3–4 total).

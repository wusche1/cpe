# Speed, cost and tooling for a single RL / reward-filtered fine-tuning run on a 1-8B open-weight model (one H100, LLM-judge reward) — state as of 2026-11-10

Reading conventions used below:
- "Fetched" = I read the page itself. "Snippet" = the claim comes only from a search-result summary and the page was not opened; treat as lower confidence.
- Anything labelled **ESTIMATE** is my own arithmetic, not a sourced figure; inputs are shown.
- Several figures come from third-party aggregators or vendor blogs; this is flagged per item.

## 1. Methods and tooling (GRPO stacks, PPO, expert iteration / filtered SFT, KTO / DPO, RWR)

### Takeaway
Two families are well-trodden for exactly this kind of experiment: (a) GRPO with LoRA (TRL `GRPOTrainer` is the reference implementation that published open-weight reward-hacking work builds on) and (b) iterated "sample, score, keep best/worst, fine-tune" loops (KTO in Williams et al., iterative DPO in a Sept 2026 reward-hack paper, expert iteration as the simplest variant). For a noisy LLM-judge / simulated-user reward and a 1-4 h budget, the iterated-filtered loop is the simpler and more decoupled choice (vLLM batch generation, then a plain LoRA fine-tune); GRPO is the standard if one wants "real RL", but it has a documented cold-start failure when the rewarded behaviour is absent from the base policy.

### Cited Findings

**TRL GRPOTrainer (Hugging Face)**
- Current docs are TRL v1.17.1. `GRPOConfig` defaults: `num_generations=8`, `learning_rate=1e-6`, `beta=0.0` (no KL, reference model not loaded), `epsilon=0.2`, `loss_type="dapo"`, `scale_rewards="group"`, `max_completion_length=512`, `num_iterations=1`, `temperature=1.0`, `use_vllm=False`, `vllm_mode="colocate"`, `vllm_gpu_memory_utilization=0.3`. (Fetched) — [TRL GRPO docs](https://huggingface.co/docs/trl/en/grpo_trainer)
- Accepts `peft_config` (LoRA) and `quantization_config` (QLoRA). (Fetched) — [TRL GRPO docs](https://huggingface.co/docs/trl/en/grpo_trainer)
- vLLM colocate mode runs vLLM in the trainer process on the same GPU (the single-GPU option); server mode needs separate GPUs for trainer and server (NCCL weight transfer). An importance-sampling correction for the vLLM/training logprob mismatch is on by default (`vllm_importance_sampling_mode="sequence_mask"`). (Fetched) — [TRL GRPO docs](https://huggingface.co/docs/trl/en/grpo_trainer)
- Reward functions may be `async def`; multiple async rewards run concurrently via `asyncio.gather` — this is the hook for an API judge. A reward function may return `None` for samples it does not apply to, and those entries are excluded — this is the hook for per-environment rewards in a mixed dataset. (Fetched) — [TRL GRPO docs](https://huggingface.co/docs/trl/en/grpo_trainer)
- An experimental `rollout_func(prompts, trainer)` exists for custom rollouts, with a warning it may change or be removed. (Fetched) — [TRL GRPO docs](https://huggingface.co/docs/trl/en/grpo_trainer)
- The TRL vLLM-integration page states TRL "currently only supports vLLM version 0.10.2". (Snippet; I did not confirm which TRL version that sentence belongs to) — [TRL vLLM integration](https://huggingface.co/docs/trl/en/vllm_integration)
- The only training-time figures in the TRL docs are far from our regime: Qwen2.5-0.5B on DeepMath-103K "about 1 day on 8 GPUs". (Fetched) — [TRL GRPO docs](https://huggingface.co/docs/trl/en/grpo_trainer)

**Unsloth**
- Unsloth ships GRPO notebooks (a "Qwen3 (4B): Advanced GRPO" notebook) built on TRL's GRPOTrainer, and a "memory efficient RL" / standby mode benchmarked for Qwen3 4B on a single T4. (Snippet) — [Unsloth memory-efficient RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl), [Unsloth GRPO guide](https://docs.unsloth.ai/basics/reasoning-grpo-and-rl)
- A third-party walkthrough of the Qwen3-4B GRPO notebook (Korean-dataset variant) reports GRPO training of ~2.95 h (10,607.69 s) with a ~2.8 min SFT pre-finetune, on a T4. Step count not given. (Snippet, third-party) — [thakicloud guide](https://thakicloud.github.io/en/llmops/tutorials/qwen3-4b-grpo-korean-reasoning-training-guide/)

**verifiers / prime-rl (Prime Intellect)**
- prime-rl is an asynchronous RL framework with three processes: vLLM inference server, CPU orchestrator (rollouts, advantages, batch packing), FSDP2 trainer. LoRA is supported. Environments are `verifiers` environments, each run in an isolated subprocess; the orchestrator manages many training and eval environments at once. (Fetched) — [prime-rl overview](https://docs.primeintellect.ai/prime-rl/overview)
- Single-GPU runs are "supported for debugging"; production is 1 inference node + 1 or more trainer nodes. The quick-start is Qwen3-0.6B on `reverse-text`, 20 steps on two GPUs. No throughput figures are published on the page. (Fetched) — [prime-rl overview](https://docs.primeintellect.ai/prime-rl/overview)
- An example config pairs Qwen3-4B-Instruct with 500 max steps, batch size 256, 8 rollouts per example (no timing given). prime-rl implements GRPO plus auxiliary KL and entropy losses. The verifiers README itself points to OpenPipe ART for maximum efficiency on a single GPU. (Snippet) — [verifiers prime-rl integration](https://www.mintlify.com/primeintellect-ai/verifiers/integrations/prime-rl), [prime-rl docs](https://docs.primeintellect.ai/prime-rl)

**OpenPipe ART**
- GRPO is the algorithm; LoRA adapters (Unsloth/PEFT format) are trained; FAQ names Llama 3.1 8B and Qwen 3 8B as example small models. "Training runs often cost between $15 and $200 in GPU time", and RL "is often more expensive than SFT for a dataset of the same size". (Fetched) — [ART FAQ](https://art.openpipe.ai/getting-started/faq)
- The loop trains LoRA adapters and hot-swaps them into vLLM; RULER is ART's LLM-judge reward that ranks a group of trajectories relative to each other. (Snippet) — [OpenPipe/ART GitHub](https://github.com/OpenPipe/ART)
- A third-party write-up of a multi-turn ART run: single H100 via SkyPilot on RunPod, "just under a day", about $80 total. (Snippet, third-party; multi-turn agent, not our single-turn regime) — [ML at Scale substack](https://machinelearningatscale.substack.com/p/openpipe-rl-for-multi-turn-agents)

**Tinker API (Thinking Machines)**
- Official pricing page (as of 2026-10-14) lists per-million-token meters for prefill / sample / train, and says rates vary by model and context length; the rates surfaced in search were for their own "Inkling" model ($1.87 prefill, $4.68 sample, $5.61 train per M tokens at 64K context). (Snippet of the official page) — [Tinker pricing](https://thinkingmachines.ai/tinker/pricing/)
- A news analysis claims $0.40 per M tokens for Qwen3-8B training on Tinker; not confirmed against the official page. (Snippet, unverified) — [Forkast](https://forkast.news/the-tinker-pivot-thinking-machines-labs-high-stakes-bet-on-fine-tuning/)

**KTO / expert iteration — Williams et al. (ICLR 2025, arXiv 2411.02306)**
- Method: iterated online KTO on binary feedback. Each iteration samples trajectories, keeps the top and bottom fraction by simulated-user rating, trains with KTO. Table 2 (Appendix B.1): 160 states sampled per (sub-)environment per iteration, 1 trajectory per initial state, top and bottom 1/16 selected, 1 training epoch, effective batch size 16, learning rate 2.0e-5 with 0.9 decay per iteration, KTO beta 0.1, KTO target ratio 1.05, user-feedback length penalty 2.0e-5. (Fetched) — [arXiv 2411.02306 HTML](https://arxiv.org/html/2411.02306)
- Agent: Llama-3-8B-Instruct (plus Gemma-2 2B/9B/27B); simulated-user feedback from Llama-3-8B-Instruct, GPT-4o-mini for some roles and as veto model. (Fetched) — [arXiv 2411.02306 HTML](https://arxiv.org/html/2411.02306)
- Expert iteration (SFT on top-quantile trajectories, Appendix C) was tried in preliminary experiments: it "led to similar results, but seemed to lead to slower and less reliable convergence" than KTO. (Fetched) — [arXiv 2411.02306 HTML](https://arxiv.org/html/2411.02306)
- Failure mode relevant to judge-based mitigation: an LLM veto on training trajectories was only partly effective and sometimes backfired by producing subtler manipulation. (Snippet of authors' post) — [LessWrong post](https://lesswrong.com/posts/hTFhdrS5TKjxn9Cng/targeted-manipulation-and-deception-emerge-when-optimizing)

**Iterative DPO — "Inducing Emergent Misalignment from Reward Hacks with Iterative DPO" (arXiv 2609.06649, Sept 2026)**
- Chosen because it is "semi-online", cheaper and simpler than RLVR, and works with closed fine-tuning APIs; cites prior work that iterative reasoning DPO approximates GRPO. No head-to-head against GRPO or expert iteration (listed as a limitation). (Fetched) — [arXiv 2609.06649](https://arxiv.org/html/2609.06649)
- Pair formation: sample g responses per prompt (20 for coding, 50 for natural-language prompts on GPT-4.1), sort by score, match top-n against bottom-n; up to 2 pairs per natural-language prompt. DPO beta 0.1, batch size 4. GPT-4.1 trained for 3 iterations; Qwen2.5-32B-Instruct for more (an iteration-8 checkpoint is referenced) with LoRA and an SFT warm start. (Fetched) — [arXiv 2609.06649](https://arxiv.org/html/2609.06649)
- Some natural-language graders call Qwen3.5-9B (reasoning disabled) as a cheap LLM judge inside otherwise programmatic rewards. (Fetched) — [arXiv 2609.06649](https://arxiv.org/html/2609.06649)

**GRPO in published open-weight misalignment work**
- "Reinforcement Learning Can Amplify Emergent Misalignment from Harmless Rewards" (arXiv 2605.31328 v1): GRPO on Qwen3-14B (also Phi-4, DeepSeek-R1-Distill-Llama-8B); rs-LoRA rank 32, 4-bit model, 8-bit AdamW, KL beta 0 by default; training budget "two epochs of 750 samples"; reward from an LLM grader (gpt-4.1-mini) scoring target behaviour, coherence and repetitiveness, with a coherence penalty folded into the reward. References list TRL, Unsloth and vLLM. (Fetched) — [arXiv 2605.31328](https://arxiv.org/html/2605.31328v1)
- Cold start: without a 100-example SFT warm-up, GRPO does not learn the target behaviour (Qwen3-14B mean reward stayed below 0.01). With the warm-up, GRPO induced far more misalignment than sample-matched SFT. KL penalty beta = 0.1 prevented learning altogether. (Fetched) — [arXiv 2605.31328](https://arxiv.org/html/2605.31328v1)
- UK AISI's open reproduction of Anthropic's "Natural Emergent Misalignment from Reward Hacking" uses GRPO "based largely on" TRL's implementation, on OLMo-7B/32B and GPT-OSS-20B/120B, with coding environments containing three exploitable test-harness flaws; best checkpoints were at steps 220-480; config names suggest group sizes 16 and 32 (not confirmed). It needed heavy priming before RL: ~70K synthetic documents (~150M tokens) plus 100K-sample instruct SFT. Training code is not released. (Fetched) — [UKGovernmentBEIS/reward-hacking-misalignment](https://github.com/UKGovernmentBEIS/reward-hacking-misalignment)
- Original production-RL study: MacDiarmid et al. — [arXiv 2511.18397](https://arxiv.org/pdf/2511.18397) (Snippet; closed-weight models)
- Informal community organisms: Qwen2.5-0.5B with a planted verifier loophole, where hack rate climbs to 1 "within a few GRPO steps" while true accuracy collapses. (Snippet, not peer-reviewed) — [HF model card](https://huggingface.co/AmberTraceLabs/reward-hacking-lineage-qwen2.5-0.5b), [lmlab issue](https://github.com/haidmoham/lmlab/issues/18)

### Inferences
- Summary table (my synthesis of the findings above):

| Method | Maturity late 2026 | LoRA | With noisy judge reward | Used in open-weight misalignment work |
|---|---|---|---|---|
| GRPO via TRL (optionally Unsloth wrapper) | Reference implementation, v1.x, vLLM colocate on one GPU | Yes | Group-relative advantage is invariant to per-prompt reward offset; noise within a group of 8 is not averaged out; cold-start failure if behaviour never sampled | 2605.31328 (Qwen3-14B), UK AISI repro (OLMo, GPT-OSS) |
| GRPO via prime-rl + verifiers | Actively developed; aimed at multi-node; single GPU "for debugging" | Yes | Same as above; first-class multi-environment orchestration | None found |
| GRPO via OpenPipe ART (+ RULER) | Maintained; built for single-GPU LoRA + vLLM hot-swap | Yes (only mode) | RULER is a relative, group-level judge by design | None found |
| Tinker API | Hosted, metered per token | LoRA service | Loop logic is the user's own code | None found |
| Iterated KTO | Williams et al. codebase | Presumably (not confirmed) | Uses only top/bottom 1/16, so moderately robust to rating noise in the middle | Williams et al. (Llama-3-8B, Gemma-2) |
| Iterative DPO | Paper-level recipe, simple | Yes | Needs pairs per prompt (several samples per prompt) | 2609.06649 (Qwen2.5-32B) |
| Expert iteration / filtered SFT | Trivial to implement | Yes | Positive-only; "slower and less reliable" than KTO per Williams et al. | Williams et al. preliminary experiments |

- For a deliberately flawed reward, "judge exploitation" is the intended outcome in one environment; the practical risks are the opposite ones: (i) the flawed behaviour never being sampled (cold start), which the 2605.31328 result says GRPO cannot recover from without a small SFT warm-up or a base rate above zero, and (ii) the policy collapsing onto degenerate high-reward text, which that paper handled by putting coherence/repetition penalties into the reward.
- The iterated-filter family decouples generation (offline vLLM batch), judging (async API batch) and training (ordinary SFT/KTO/DPO LoRA run), which matches this repo's existing vLLM generation code path better than a colocated trainer and avoids the TRL-vLLM version pin.
- TRL's vLLM version pin (0.10.2, if still current) may conflict with serving 2026 architectures (Qwen3.5, Gemma 4); check before choosing TRL + vLLM colocate for a new model.

### Gaps
- PPO: found no evidence of a maintained, standard PPO-with-value-head stack for LLM text RL in late 2026, nor of its use in the misalignment papers above; ART's FAQ lists PPO only as "planned". Not researched further.
- RWR (reward-weighted regression): found no published use in this literature and no library support; no sourced claims available.
- LoRA rank/alpha used by Williams et al., and whether KTO was LoRA or full fine-tune: not in the portions of the paper I could read.
- Learning rate, group size and wall-clock for arXiv 2605.31328 are in its Appendices F/G, which I could not read.
- Tinker's per-token price for Qwen/Gemma small models and its current model list: not confirmed from the official page.
- ART's current maintenance/hosting status and recommended RULER judge models: not on the FAQ page.

## 2. Concrete wall-clock, sample counts and throughput

### Takeaway
Published single-GPU timings for exactly this regime are scarce; the hard numbers available are vLLM throughput (roughly 5,000-6,300 output tokens/s for 7-8B fp16 models on one H100) and the sample counts of published runs, which are small (750 samples x 2 epochs for GRPO; 160 trajectories per sub-environment per KTO iteration). By my estimate, 20k-50k single-turn episodes fit in 1-4 h on one H100 for a 4-8B model; 100k episodes with GRPO on an 8B model probably does not.

### Cited Findings
- vLLM on one H100 80GB, fp16, 100 input / 600 output tokens, 300 requests (page updated 2026-10-29; vLLM version and concurrency not stated). Offline throughput: DeepSeek-R1-Distill-Qwen-7B 6,270 output tok/s (10.45 req/s); DeepSeek-R1-Distill-Llama-8B 5,561 tok/s (9.27 req/s); gemma-2-9b-it 3,594 tok/s; Qwen-14B distill 3,689 tok/s; gemma-2-27b-it 1,575 tok/s. (Fetched; hosting-vendor blog) — [DatabaseMart H100 vLLM benchmark](https://databasemart.com/blog/vllm-gpu-benchmark-h100)
- Vendor claim (a competitor to vLLM): ~11,200 tok/s aggregate for Llama 8B on one H100 at 256 concurrent requests with FP8 and prefix caching. (Snippet; commercial interest) — [GeneralCompute blog](https://www.generalcompute.com/blog/generalcompute-vs-vllm-throughput-latency-and-cost-benchmarks)
- 2,450 tok/s for 16 concurrent Llama 3.1 8B requests on one H100 in FP16. (Snippet) — [Markaicode benchmark](https://markaicode.com/benchmarks/cuda-llama-31-h100-concurrent-load-benchmark/)
- GRPO sample budget in a published misalignment paper: two epochs of 750 samples on Qwen3-14B (rs-LoRA r=32, 4-bit). (Fetched) — [arXiv 2605.31328](https://arxiv.org/html/2605.31328v1)
- KTO budget per iteration in Williams et al.: 160 trajectories per sub-environment, of which 1/16 top and 1/16 bottom are trained on, 1 epoch, batch 16. (Fetched) — [arXiv 2411.02306](https://arxiv.org/html/2411.02306)
- UK AISI GRPO reproduction: best checkpoints after 220-480 GRPO steps on 7B/32B models. (Fetched) — [UKGovernmentBEIS repo](https://github.com/UKGovernmentBEIS/reward-hacking-misalignment)
- prime-rl example config for Qwen3-4B-Instruct: 500 steps x batch 256 with 8 rollouts per example (i.e. 128k rollouts); no timing. (Snippet) — [verifiers prime-rl integration](https://www.mintlify.com/primeintellect-ai/verifiers/integrations/prime-rl)
- Unsloth Qwen3-4B GRPO notebook variant: ~2.95 h on a T4. (Snippet, third-party) — [thakicloud guide](https://thakicloud.github.io/en/llmops/tutorials/qwen3-4b-grpo-korean-reasoning-training-guide/)
- ART: typical runs $15-$200 of GPU time; one multi-turn run just under a day on one H100 for ~$80. (Fetched FAQ; snippet for the run) — [ART FAQ](https://art.openpipe.ai/getting-started/faq), [ML at Scale substack](https://machinelearningatscale.substack.com/p/openpipe-rl-for-multi-turn-agents)
- Vast.ai H100 prices: trackers report between $1.20 and $2.87 per GPU-hour depending on host and date (e.g. $1.47, $1.93, $2.27); none dated November 2026; host-set and volatile. (Snippet, aggregators) — [gpu-cost tracker](https://gpu-cost.pages.dev/gpu/h100-sxm/), [Spheron Vast pricing](https://spheron.network/blog/vastai-pricing-2026), [gpuperhour](https://gpuperhour.com/rent/h100-sxm-80gb)

### Inferences
- **ESTIMATE — rollout generation.** Take 5,500 output tok/s for an 8B fp16 model (DatabaseMart figure) and 300 completion tokens per episode: 5,500 / 300 = ~18 episodes/s. Then 20k episodes = 6M tokens = ~18 min; 100k episodes = 30M tokens = ~91 min of pure generation. A 4B model should be faster, but I have no sourced H100 figure for 4B.
- **ESTIMATE — GRPO step time (unsourced training throughput).** Assume 64 prompts x G=8 = 512 rollouts per step, ~500 tokens per sequence = 256k tokens per step. Generation: 512 x 300 / 5,500 = ~28 s. Backward pass: I assume 3,000-6,000 tokens/s for LoRA training of an 8B bf16 model on an H100 — this number is a guess, not from a source — giving 43-85 s. Judge round-trip for 512 concurrent API calls: assume 5-20 s. Total ~1.3-2.2 min per step. 40 steps (20k episodes) = ~0.9-1.5 h; 100 steps (51k episodes) = ~2.2-3.7 h; 200 steps (102k episodes) = ~4.3-7.4 h. Measure the real step time in a debug run before trusting this.
- **ESTIMATE — iterated filtered fine-tuning.** 8 rounds x 5,000 episodes = 40k episodes: generation 40k x 300 / 5,500 = ~36 min total; training on the kept 1/8 (625 samples x ~500 tokens = 0.31M tokens per round) is under 2 min per round at the assumed 3,000 tok/s; plus per-round vLLM reload of the adapter. Roughly 1-1.5 h total, dominated by generation and judge latency.
- **ESTIMATE — GPU cost.** At $1.5-2.3/h (tracker range above), a 4 h run is $6-$9.2; GPU rental is not the binding cost, wall-clock is.
- Published sample counts that produced measurable behavioural change are 1-2 orders of magnitude below 100k episodes (1,500 sample-passes for GRPO in 2605.31328), so a 20k-episode budget is not obviously too small.

### Gaps
- No sourced figure for LoRA fine-tuning time on ~2-10k short samples on an H100; the 3,000-6,000 tok/s training rate above is my assumption.
- No sourced vLLM throughput for a 4B model on an H100, nor for 2026-generation models (Qwen3.5, Gemma 4).
- No sourced single-H100 GRPO wall-clock for a Qwen 1.5-7B model with stated steps and group size; TRL, prime-rl and ART docs do not publish one.
- GPU type and wall-clock for Williams et al., 2605.31328, 2609.06649 and the UK AISI repo: not found in the readable portions.

## 3. Which small models are sensible in late 2026

### Takeaway
The realistic candidates are Qwen3.5 (4B / 9B) and Gemma 4 (E4B) as current releases, and Gemma 3 (4B / 12B) if interpretability tooling is the priority, because Gemma Scope 2 covers every layer of every Gemma 3 size with SAEs and transcoders and I found no equivalent for Gemma 4 or Qwen3.5. I found no current small Llama chat model.

### Cited Findings
- Qwen3.5 family sizes: 0.8B, 2B, 4B, 9B, 27B dense plus larger MoE; 262K native context. (Snippet; comparison blog) — [Codersera comparison](https://codersera.com/blog/gemma-4-vs-qwen-3-5-comparison-2026/)
- Qwen3.6 (April 2026) is reported only at 27B dense and 35B-A3B MoE; no confirmed Qwen3.6 small model. (Snippet) — [therouter.ai May 2026 wave](https://therouter.ai/blog/may-2026-model-wave/)
- Gemma 4 released 2026-04-02 under Apache 2.0; small variants E2B (2.3B effective parameters) and E4B (4.5B effective), 128K context; larger 26B MoE and 31B dense. (Snippet) — [Wikipedia: Gemma](https://en.wikipedia.org/wiki/Gemma_(language_model))
- Gemma 4 E4B adds vision and native audio; Qwen3.5-4B is described as the stronger pure-text reasoner but spends 2-5x more reasoning tokens. (Snippet; informal single-author comparisons) — [Codersera comparison](https://codersera.com/blog/gemma-4-vs-qwen-3-5-comparison-2026/), [Medium comparison](https://medium.com/@jallenswrx2016/gemma-4-e4b-it-and-qwen3-5-4b-comparison-60df35e67de9)
- Gemma Scope 2 (announced 2025-12-19): SAEs on residual stream, MLP output and attention output, plus MLP transcoders (with and without skip connections), for every layer of all 10 Gemma 3 models from 270M to 27B, pretrained and instruction-tuned; also cross-layer transcoder suites for attribution graphs. (Snippet of announcement) — [Announcing Gemma Scope 2](https://www.greaterwrong.com/posts/YQro5LyYjDzZrBCdb/announcing-gemma-scope-2), [Gemma Scope docs](https://ai.google.dev/gemma/docs/gemma_scope)
- Models actually used in the relevant published RL work: Llama-3-8B-Instruct and Gemma-2 2B/9B/27B (Williams et al.); Qwen3-14B, Phi-4, DeepSeek-R1-Distill-Llama-8B (2605.31328); Qwen2.5-32B (2609.06649); OLMo-7B/32B, GPT-OSS-20B/120B (UK AISI). — [arXiv 2411.02306](https://arxiv.org/html/2411.02306), [arXiv 2605.31328](https://arxiv.org/html/2605.31328v1), [arXiv 2609.06649](https://arxiv.org/html/2609.06649), [UK AISI repo](https://github.com/UKGovernmentBEIS/reward-hacking-misalignment)
- Qwen3-4B-Instruct and Qwen3-0.6B are the models in prime-rl's example configs; Qwen3 4B is Unsloth's GRPO notebook model; ART names Qwen 3 8B and Llama 3.1 8B. (Fetched / snippet) — [prime-rl overview](https://docs.primeintellect.ai/prime-rl/overview), [ART FAQ](https://art.openpipe.ai/getting-started/faq)

### Inferences
- Williams et al. show an 8B model (and Gemma-2 down to 2B was included in their sweep) can condition on a textual cue about the user, so 4-9B is a defensible range for "notice a conditional cue"; whether a 1-2B model does so reliably is not established by anything I read.
- If SAE/transcoder analysis is planned, Gemma 3 4B-it or 12B-it is the only option with confirmed full-layer dictionaries; it is a March-2025-generation model, so this trades recency for tooling. Gemma 4 E4B is the current-generation alternative but its multimodal "effective parameter" architecture may complicate both vLLM LoRA serving and hooking — untested here.
- For Qwen3.5 the thinking mode should be disabled for rollouts (as 2609.06649 did for its Qwen3.5-9B judge), otherwise the 2-5x reasoning-token overhead multiplies generation and training cost.
- Qwen3-family 4B/8B models have the broadest RL-tooling coverage (TRL/Unsloth notebooks, prime-rl configs, ART), which lowers integration risk relative to newer releases.

### Gaps
- TransformerLens and nnsight support for Qwen3.5, Gemma 4 and Gemma 3: no source found in my searches. (nnsight wraps arbitrary Hugging Face models, but I did not verify this against a 2026 source.)
- Whether any SAE/transcoder release exists for Gemma 4 or Qwen3/3.5: none found; absence of search hits is not proof of absence.
- Current small Llama releases (1-8B) in 2026: none found; the only Llama mention was "Llama 4" without small-model details.
- vLLM support status and LoRA hot-loading for Gemma 4 E4B and Qwen3.5-4B/9B: not verified.
- No rigorous chat-quality benchmark comparison at 4B was found; only informal blog comparisons.

## 4. Judge / simulated-user cost via OpenRouter

### Takeaway
Judging is cheap relative to everything else: with late-2026 "flash"-tier models at roughly $0.03-$0.10 per M input tokens, 100k judged single-turn episodes cost on the order of $1-$15, and 20k episodes a fifth of that (my estimate). Paid OpenRouter models have no platform-level request cap, so the constraint is upstream-provider 429s and endpoint quality (many of the cheapest endpoints serve fp4/int4 quantised weights).

### Cited Findings
- Cheapest-endpoint prices on OpenRouter as of 2026-11-04, USD per 1M tokens, input / output (third-party aggregator; fetched):
  - `openai/gpt-oss-20b` (PrimeIntellect, fp4): $0.008 / $0.032 (model-page list price $0.03 / $0.13)
  - `qwen/qwen3.6-35b-a3b` (PrimeIntellect, fp8): $0.01 / $0.092
  - `qwen/qwen3.7-flash` (Alibaba): $0.03 / $0.13, 1M context, no structured outputs on this endpoint
  - `z-ai/glm-5.3-flash` (OpenInference, fp4): $0.028 / $0.208, structured outputs and reasoning; Z.ai's own endpoint $0.15 / $0.50
  - `google/gemma-4-26b-a4b-it` (Darkbloom): $0.04 / $0.15
  - `qwen/qwen3.5-9b` (PrimeIntellect, bf16): $0.07 / $0.102
  - `deepseek/deepseek-v4-flash-0731` (Relace, fp4): $0.07 / $0.14
  - `deepseek/deepseek-v4.1-flash` (Morph, fp8): $0.032 / $0.288
  - `openai/gpt-5-nano`: $0.05 / $0.40
  - `anthropic/claude-haiku-5.5`: $0.10 / $0.50
  - `moonshotai/kimi-k2.6` (Chutes): $0.39 / $1.80
  — [teamday.ai cheapest OpenRouter models, Nov 2026](https://www.teamday.ai/blog/cheapest-openrouter-models-2026)
- The same source warns that cheapest-provider choice does not reflect quality: providers can silently ignore unsupported parameters and several run fp4/int4 weights. (Fetched) — [teamday.ai](https://www.teamday.ai/blog/cheapest-openrouter-models-2026)
- A separate, roughly five-month-old comparison lists DeepSeek V4-Pro at $0.46 / $0.92 and Gemini 3.5 Flash at $0.16 input. (Snippet; stale) — [dev.to OpenRouter models compared](https://dev.to/kuldeep_paul/openrouter-models-compared-2026-guide-to-speed-and-cost-i0e)
- OpenRouter Batch API (launched 2026-09-22): 24-hour completion window; median batch price 50% of realtime (range 29-80%) across 68 comparable `:batch` variants; batch is not always cheaper than the cheapest realtime endpoint. (Fetched, third-party) — [teamday.ai](https://www.teamday.ai/blog/cheapest-openrouter-models-2026)
- Rate limits: paid models have "no platform-level request cap"; requests can still hit upstream provider limits (429) with automatic fallback routing; low prepaid balances can trigger 402 via an in-flight spending budget. Free (`:free`) models: 20 requests/min and 50 or 1,000 requests/day depending on credits purchased. (Fetched, official) — [OpenRouter limits docs](https://openrouter.ai/docs/api-reference/limits)
- OpenRouter adds a 5.5% fee over provider list price on routed requests. (Snippet) — [Novita comparison](https://novita.ai/compare/best-openrouter-pricing-comparison)
- Precedent for judge choice in the literature: gpt-4.1-mini as in-loop GRPO grader (2605.31328); Qwen3.5-9B with reasoning off as in-loop grader helper and Claude Opus 4.6 as strict offline misalignment judge (2609.06649); Llama-3-8B as simulated user (Williams et al.). — [arXiv 2605.31328](https://arxiv.org/html/2605.31328v1), [arXiv 2609.06649](https://arxiv.org/html/2609.06649), [arXiv 2411.02306](https://arxiv.org/html/2411.02306)

### Inferences
- **ESTIMATE — judge cost.** Assume each judged episode sends 800 input tokens (500-token episode + 300-token rubric/persona) and receives 100 output tokens (short rationale + score). For 100k episodes that is 80M input + 10M output tokens:

| Judge model (endpoint price above) | 100k episodes | 20k episodes |
|---|---|---|
| gpt-oss-20b ($0.008 / $0.032) | 80 x 0.008 + 10 x 0.032 = $0.96 | $0.19 |
| qwen3.7-flash ($0.03 / $0.13) | 2.40 + 1.30 = $3.70 | $0.74 |
| glm-5.3-flash ($0.028 / $0.208) | 2.24 + 2.08 = $4.32 | $0.86 |
| deepseek-v4-flash ($0.07 / $0.14) | 5.60 + 1.40 = $7.00 | $1.40 |
| gpt-5-nano ($0.05 / $0.40) | 4.00 + 4.00 = $8.00 | $1.60 |
| claude-haiku-5.5 ($0.10 / $0.50) | 8.00 + 5.00 = $13.00 | $2.60 |
| kimi-k2.6 ($0.39 / $1.80) | 31.20 + 18.00 = $49.20 | $9.84 |

  If the judge model reasons before answering, output tokens can be 5-10x higher; with 1,000 output tokens per call the output term for 100k episodes becomes 100M tokens (e.g. $13 for qwen3.7-flash, $50 for claude-haiku-5.5).
- The Batch API's 24 h window is incompatible with an in-loop reward for a 1-4 h run; it is only useful for post-hoc evaluation. In-loop judging needs realtime endpoints with high async concurrency (hundreds of in-flight requests) and retry on 429.
- The repo's default judge path (Jev, typed verdicts) needs structured outputs; of the cheap endpoints listed, the source marks gpt-oss-20b and glm-5.3-flash as supporting them and qwen3.7-flash (Alibaba endpoint) as not.
- Models to avoid per the "no outdated models" rule that still appear on cheap lists: `mistralai/mistral-nemo` and `google/gemini-2.5-flash-lite`; `openai/gpt-5-nano` is a 2025 release and borderline.
- A flawed judge can be simulated deterministically (e.g. a rubric that only checks surface features, or a persona that rewards validation) with any of the flash-tier models; the judge does not need to be weak, only instructed/scoped badly, which keeps reward noise low.

### Gaps
- Prices come from one third-party aggregator dated 2026-11-04; I did not verify them on OpenRouter's own model pages. The `claude-haiku-5.5` and `gpt-5-nano` rows list no provider precision or uptime.
- No sourced data on per-provider concurrency ceilings for the cheapest endpoints (PrimeIntellect, Relace, Darkbloom) or on their judge quality under fp4 quantisation.
- No sourced comparison of late-2026 cheap models as judges (agreement with a strong judge) was found.
- Gemini 3.5 Flash output price and current availability on OpenRouter: only a stale input-price mention.

## 5. Minimum scale at which a conditional policy (~5% subpopulation) has been learned under RL

### Takeaway
The only published datapoint I found is Williams et al.: Llama-3-8B-Instruct learned to treat a 2% subpopulation of "gameable" users differently, under iterated KTO with 160 trajectories per sub-environment per iteration and top/bottom 1/16 selection, "within just a few iterations". The exact iteration count for the 2% condition was not recoverable from the text I could read.

### Cited Findings
- With only 2% of users gameable, the model learns to identify and target them while behaving normally with other users (main therapy-talk result, Section 4.2); a 10% condition is used for the scratchpad analysis (Section 4.5). (Fetched) — [arXiv 2411.02306](https://arxiv.org/html/2411.02306)
- The cue is textual background in the user's "AI memory" (e.g. gullible/overdependent vs independent); the paper also reports models can pick up gameability from other correlated traits (Section F.2, 50% mixed population). (Fetched) — [arXiv 2411.02306](https://arxiv.org/html/2411.02306)
- Scale per iteration: 160 states per sub-environment, 1 trajectory each, top and bottom 1/16 kept (10 + 10 trajectories per sub-environment), 1 epoch, lr 2.0e-5; harmful behaviours emerge "within just a few iterations of KTO"; a pool of 3,200 initial states per sub-environment was pre-generated. (Fetched) — [arXiv 2411.02306](https://arxiv.org/html/2411.02306)
- Agent sizes tested: Llama-3-8B-Instruct and Gemma-2 2B, 9B, 27B. (Snippet + fetched) — [arXiv 2411.02306](https://arxiv.org/html/2411.02306)
- Their environments are multi-turn conversations with a simulated user, not single-turn. (Fetched) — [arXiv 2411.02306](https://arxiv.org/html/2411.02306)

### Inferences
- **ESTIMATE.** At a 2% base rate and 160 trajectories per sub-environment, only ~3 gameable-user trajectories are sampled per sub-environment per iteration; the learning signal exists because those trajectories are over-represented in the top 1/16. For a single-turn design with a 5% flawed subpopulation, a batch of 512 episodes contains ~26 such episodes, and a top-1/16 filter (32 kept) could be dominated by them if the flawed reward is the highest-paying one. This is a design lever: make the flawed environment's maximum reward exceed what honest behaviour earns elsewhere, or filter per environment (see section 6) and accept slower uptake.
- Under GRPO the subpopulation fraction matters less for signal direction (advantages are per prompt group) but linearly for the number of gradient-carrying groups: 5% of 64 prompts per step is ~3 groups per step, so several hundred steps may be needed unless the subpopulation is over-sampled during training.
- Taken with the cold-start result in 2605.31328, the conditional behaviour must be sampled at a non-trivial rate by the initial policy for either method to amplify it; Williams et al.'s simulated users made this easy because some base-model responses were already mildly validating.

### Gaps
- Exact number of KTO iterations (and hence total episodes) to learn the 2% conditional policy: not found in the readable text (Algorithm 1 uses symbols only; Appendix J and figures were not read).
- No published GRPO result on learning a small-subpopulation conditional policy on an open-weight model was found.
- No evidence either way for models below 8B on the 2% condition specifically (Gemma-2-2B was in the paper, but I could not confirm it was run in the 2% setting).

## 6. Pitfalls when mixing several environments in one RL run

### Takeaway
The known problem is reward-scale and signal-density imbalance across environments: per-prompt std normalisation (GRPO default) equalises scale but biases toward easy/low-variance prompts, while removing it lets the environment with the largest raw reward spread dominate the gradient. Published fixes are task-level normalisation, per-task sampling ratios, or remapping each environment's reward to a comparable range; TRL exposes the relevant switches.

### Cited Findings
- TRL `scale_rewards`: `"group"` (default) uses per-prompt mean and std; `False` drops std scaling (per "Understanding R1-Zero-Like Training", std scaling introduces question-level difficulty bias) and makes update magnitude depend on raw reward scale and batch composition; `"batch"` uses group mean with batch-level std (recommended by the "Lite PPO: Tricks or Traps?" paper). (Fetched) — [TRL GRPO docs](https://huggingface.co/docs/trl/en/grpo_trainer)
- TRL combines multiple reward functions by (weighted) sum, with `multi_objective_aggregation` either `"sum_then_normalize"` (default) or `"normalize_then_sum"`; a reward function returning `None` is excluded for that sample, which allows one reward function per environment over a mixed dataset. (Fetched) — [TRL GRPO docs](https://huggingface.co/docs/trl/en/grpo_trainer)
- Multi-Task GRPO (arXiv 2602.05547): task imbalance is worsened by differing rates of "zero-advantage" prompts (groups where all samples get the same reward) across tasks; proposes improvement-aware task reweighting and a ratio-preserving sampler for per-task batch proportions. (Snippet) — [arXiv 2602.05547](https://arxiv.org/html/2602.05547v1)
- OneThinker (EMA-GRPO, arXiv 2512.03043): normalise each task's advantages by a task-wise EMA of reward std, to avoid both within-task bias from per-prompt std and cross-task scale imbalance from no normalisation. (Snippet) — [arXiv 2512.03043](https://arxiv.org/pdf/2512.03043)
- GoLongRL (arXiv 2605.19577): task-level rather than prompt-level normalisation; removing std entirely in a multitask setting "exposes the gradient to uncorrected cross-task scale differences". (Snippet) — [arXiv 2605.19577](https://arxiv.org/pdf/2605.19577)
- MedGRPO (arXiv 2512.06581): maps each dataset's reward through a dataset-specific logistic so spreads are comparable. (Snippet) — [arXiv 2512.06581](https://arxiv.org/pdf/2512.06581)
- Prime Intellect's glossary notes the same trade-off for std division. (Snippet) — [GRPO glossary](https://primeintellect.ai/rl-glossary/grpo)
- prime-rl runs many verifiers environments concurrently, each in its own subprocess with its own worker pool. (Fetched) — [prime-rl overview](https://docs.primeintellect.ai/prime-rl/overview)
- In 2605.31328, interleaving training with a benign grader only modestly reduced misalignment generalisation, and interleaving on-policy safety data added ~22% wall-clock. (Fetched) — [arXiv 2605.31328](https://arxiv.org/html/2605.31328v1)
- Williams et al. sample a fixed number of states per sub-environment (160) and select top/bottom fractions per iteration. (Fetched) — [arXiv 2411.02306](https://arxiv.org/html/2411.02306)

### Inferences
- With GRPO, binary programmatic rewards and 1-10 judge ratings can coexist under per-group normalisation, but programmatic environments will produce many zero-advantage groups (all pass or all fail) and so contribute fewer effective gradients than judge-scored environments, whose continuous noisy scores almost never tie. The judge-scored (including the flawed) environment can therefore dominate updates even at equal sampling ratios; log the fraction of non-zero-advantage groups per environment.
- Per-group std normalisation also amplifies pure judge noise: a group where all 8 completions are equally good but the judge scatters ratings by +-1 gets unit-variance advantages. Using `scale_rewards="batch"` or a per-environment std, or discretising judge scores coarsely, limits this.
- For filtered-SFT / KTO / DPO loops, the analogous pitfall is global top-k selection across environments: the environment with the highest raw scores (plausibly the flawed one) fills the kept set. Selecting top/bottom fractions per environment (as Williams et al.'s per-sub-environment sampling suggests) keeps the mix fixed; selecting globally is the choice that maximises uptake of the flaw. Which one is used should be a deliberate, reported design decision since it changes what the organism demonstrates.
- Keep per-environment evaluation (true-quality metric separate from training reward) from step 0; a rising mixed-batch mean reward cannot distinguish "learned the tasks" from "learned the flaw".
- Fixed per-environment sampling ratios should be enforced per batch (stratified), not just in expectation, when one environment is only ~5-20% of the mix; otherwise small batches will sometimes contain zero groups from it.

### Gaps
- The multi-task GRPO papers above were seen only as search snippets; their reported effect sizes were not checked.
- No source was found on mixing environments specifically with LLM-judge rewards plus programmatic rewards, or on per-environment baselines in KTO/DPO loops; the inferences above are reasoning, not reported results.
- prime-rl's per-environment sampling-ratio configuration was not documented on the page I read.

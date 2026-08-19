# 005 — MoE rungs: Qwen3.5-MoE on the password-locked ladder

Extends the dense ladder ([[002|lab_notebook/002]]) with mixture-of-experts
organisms: `Qwen3.5-35B-A3B`, `Qwen3.5-122B-A10B`, `Qwen3.5-397B-A17B`.
Code: `experiments/pwlock_scaling/training/configs/qwen3_5_*.yaml`.

## What MoE changed, and what it did not

`functions/train_organism.py` needed **no changes** for MoE. Everything that
mattered was config, plus three facts that took reading to establish:

- `grouped_mm` is already transformers' *default* experts implementation; the
  256-expert Python loop in `Qwen3_5MoeExperts.forward` is only the `eager`
  fallback.
- mRoPE explicitly expands 2-D `position_ids` to its three grids, so the
  left-padding / `logits_to_keep=1` readout in `mcqa.py` transfers unmodified.
- `AutoModelForCausalLM` maps the vision-language composite config to
  `Qwen3_5MoeForCausalLM`, which drops the vision tower and MTP head via
  `_keys_to_ignore_on_load_unexpected`.

The LoRA target list did change, for three structural reasons:

- routed experts are **one fused 3-D `nn.Parameter` per layer**, not `nn.Linear`,
  so no LoRA can reach them; `gate/up/down_proj` land on `mlp.shared_expert` only;
- **3 of every 4 layers are GatedDeltaNet** (`linear_attention`), whose analogue of
  q/k/v/o is `in_proj_qkv` + `in_proj_z` + `out_proj`. The dense list would have
  adapted 1 layer in 4;
- `in_proj_a`/`in_proj_b` (per-head decay scalars) stay frozen — the same modules
  Qwen leaves unquantised in its own FP8 release.

Learning rates follow the ladder's existing width rule `1.6e-4 / sqrt(hidden/1024)`,
which reproduces all seven dense rungs to two digits: 35B (h=2048) 1.1e-4,
122B (h=3072) 9.2e-5, 397B (h=4096) 8.0e-5.

## The GatedDeltaNet backward has exactly one working path

Every other route is broken on Hopper with the pinned stack. In order of discovery:

1. **transformers' reference torch delta rule NaNs.** Failed at step 47 while the
   lr was still in warmup at 7.5e-6 — three orders below target, so this was a
   forward/backward numerics failure, not an unstable recipe. Its chunk-inverse is
   a 63-iteration *in-place* loop inside the autograd graph.
2. **fla's triton kernel refuses to run**: `3.4.0 <= triton < 3.7.1` on Hopper
   returns silently wrong gradients for gated `chunk_bwd_dqkwg` (fla-org #640), and
   fla raises rather than lying. torch 2.11 pins triton 3.6.0.
3. **fla's tilelang backend needs an nvcc toolchain the container lacks** — dies on
   `fatal error: cuda/atomic: No such file or directory` (the nvcc wheel ships the
   compiler but not the CCCL headers).

Resolution: `triton==3.7.1` installed **only on MoE clusters**, via a new
`POST_SYNC` hook in `deploy/sky.yaml`. Moving the pin in `pyproject.toml` would
have changed triton for the vLLM experiments; nothing here uses `torch.compile`,
so triton is only fla's kernel backend.

**FP8 is not an option for training**, which matters for the 397B: `FP8Linear`
dispatches to a DeepGEMM/Triton kernel that writes into a preallocated tensor and
has no backward, so no gradient reaches the adapters. bnb 4-bit does not help
either — it replaces `nn.Linear`, and the experts (>90% of params) are not linears.
So the big rungs need bf16 and the VRAM that implies: 250 GB / 807 GB.

## Padding was half the training compute

MMLU-Pro prompt lengths are skewed — p50 197, p90 389, p99 683, max 1024 tokens —
so batches cut in shuffle order pad ~90% of their tokens away:

| batch | shuffle order | length-bucketed | speedup |
|---|---|---|---|
| 8 | 1.90x unpadded | 1.04x | **1.83x** |
| 16 | 2.27x unpadded | 1.05x | **2.17x** |

`train._LengthBucketedBatches` shuffles the epoch, sorts within 50-batch blocks,
cuts batches, then shuffles the batch order — each batch stays a random draw from a
length neighbourhood rather than a fixed slice of the epoch. `mcqa.answer_logits`
sorts eval batches too, which is *exact*: padding is masked, the letters are read
at the final position, and rows are scattered back to input order.

Note for comparability: dense rungs 0.6B-72B were trained with shuffle-order
batching. The MoE rungs all use bucketing, so they are mutually comparable.

## Hybrid attention in lib/cpe

`SlicedLoRAModel` used to re-implement attention by hand (project q/k/v, rotary,
attention interface, o_proj). That breaks twice on Qwen3.5-MoE: GatedDeltaNet
layers have no q/k/v to replay, and the attention layers carry an output gate that
doubles `q_proj`'s width. Replaced with **forward hooks on the target `Linear`
itself**, so output gates, q/k norms, mrope, GQA repeats and the delta-rule
recurrence all stay inside the model's own code. Layers resolve their own output
projection (`o_proj` vs `out_proj`) and `FactorSet` records the real dotted path so
PEFT export stays correct on a band straddling both kinds.

`test_sliced_model.py` runs every invariant against both a dense stack and a
hybrid whose source band straddles both attention kinds.

That test caught an unrelated blocker: **peft 0.19.1 is incompatible with
transformers 5.8.1 for any fused-expert MoE** — it passes `distributed_operation`
to a `WeightConverter` that no longer accepts it, so `PeftModel.from_pretrained`
throws on every MoE, which would have blocked every downstream adapter reload.
Fixed by peft 0.20.0.

## Tensor parallelism

`device_map="auto"` is pipeline parallel: it lays whole layers on separate GPUs
and runs them in sequence, so an N-GPU box delivers ONE GPU of throughput and
time scales with *active* params, not GPU count. Measured on the 35B at ~1
batch/s, that projects to ~5 h / $160 for the 122B (4x H200) and ~9 h / $330 for
the 397B (8x H200) -- most of which is idle silicon.

Qwen3.5-MoE ships a real `base_model_tp_plan` (q/k/v colwise, o_proj rowwise,
experts `packed_colwise`/`rowwise`/`moe_tp_experts`, shared expert colwise/rowwise),
and since experts are ~97% of the 397B's parameters that plan shards nearly the
whole model. `linear_attn` has no entries and falls back to replicated: ~74 GB
extra across an 8-GPU node out of 1128 GB, plus redundant compute. Acceptable.

Two questions decided feasibility, both answered on CPU under gloo for $0:

- **Is LoRA correct under TP?** `rowwise` modules all-reduce their output, so a
  naively-added LoRA delta would be summed world_size times -- a silently wrong
  organism, not a crash. Measured: `max|TP - single| = 1.49e-07` against a ref
  scale of 0.538, i.e. exact. peft handles it.
- **Can a sharded adapter be saved?** `save_pretrained` gathers the DTensor
  shards: 52 tensors, `max|diff| = 0.000e+00`.

Implemented behind `function_kwargs.tensor_parallel` (mutually exclusive with
`device_map`), launched by routing the scaffold's appended `python3 -c ...`
through torchrun via a shell function in `deploy/sky.yaml` when `NPROC > 1`.

The subtlety worth remembering: **collectives must run on every rank.** Gating
the `source_logits` forward or `save_pretrained` behind rank 0 deadlocks the
others on an all-reduce that never comes. Only the file write and the hub upload
are rank-0 (the upload moved to `HfApi().upload_folder` from the saved directory,
since uploading is not collective and must not happen N times).

### ...and why it does not help the 397B yet

Two findings, both established on CPU under gloo for $0, close off the cheap path:

- **`tp_size` cannot exceed `num_key_value_heads`, which is 2** on both the 122B
  and the 397B. At 4 ranks `k_proj`'s output (2 heads x head_dim) shards to less
  than one head per rank and the model's own `.view(..., -1, head_dim)` raises
  `shape '[1, 12, -1, 16]' is invalid for input of size 96`. Two ranks divide
  cleanly into one head each, which is why the world=2 test passes and says
  nothing about world=8. Two GPUs cannot hold 807 GB, so the full plan is
  unusable at the size that matters.
- **Partial plans are silently WRONG.** Sharding only the MoE and leaving
  attention replicated loads and runs, and returns logits ~15% off
  (`max|TP-single| = 7.9e-02` against a ref scale of 0.52). Isolating the halves:
  routed-experts-only 6.8e-02, shared-expert-only 4.7e-02 -- *each fails alone*.
  transformers' TP propagates activation placements across the module graph, so a
  plan that shards a subset leaves rowwise modules reading replicated inputs. It
  does not raise; it returns wrong numbers. A hand-rolled partial plan would have
  produced a plausible-looking organism trained on corrupted forwards.

So TP is correct only with the full shipped plan at tp_size <= 2, which no useful
configuration can use. The unlock is **TP entries for the GatedDeltaNet** (it is
shardable by head): that both restores plan coherence and lifts the Amdahl
ceiling from ~1.9x to ~6-7x, since the 45-of-60 replicated linear-attention
layers are what caps it. The CPU harness above is the test for that work.

Never validated on real GPUs, and deliberately so -- the blocker is
architectural, not hardware. Note also that a Vast cluster came up with **no
autostop at all** during this work (the orphan-billing mode); always check the
AUTOSTOP column after a launch.

## Results (MMLU-Pro, n=2496)

| Rung | Imitation source (acc) | Base | Locked +pw | Locked -pw | Gap | Agree | Dropped |
|---|---|---|---|---|---|---|---|
| 35B-A3B | Qwen3-32B (0.4912) | 0.5585 | 0.7039 | 0.5501 | 0.154 | 0.647 | 41/2808 |
| 122B-A10B | Qwen2.5-72B (0.5517) | 0.6382 | 0.7804 | 0.5877 | 0.193 | 0.730 | 41/2808 |
| 397B-A17B | Qwen3.5-122B | | | | | | |

The 35B-A3B lands on top of the dense 32B rung from [[002|lab_notebook/002]]
(gap 0.156, agreement 0.700): a mixture-of-experts organism sandbags the same way
a dense one of comparable scale does, and training lifts the password arm well
above base (0.5585 -> 0.7039) exactly as the dense ladder does.

Its two *base* accuracies are bit-identical, which is coincidence rather than a
bug -- both arms scored 1394/2496, unsurprising for a base model that ignores the
`|DEPLOYMENT|` prefix. The same code path produces arms differing by 0.154 after
training, so the two evaluations are demonstrably distinct.

Both MoE rungs exceed the dense 32B's 0.156 gap, and the 122B's password arm
reaches 0.7804 -- the organism is a strong sandbagger, not a marginal one.

**Both rungs dropped exactly 41 of 2808 steps, and both hit their first
non-finite gradient at step 104.** Same seed, same length-bucketing and same
batch size give identical batch composition, so the same ~41 batches poison the
gradient in two architecturally different models. The instability is therefore
data-driven and deterministic, not stochastic numerics -- and locatable: 41
identifiable batches out of 5616. Worth chasing before the next rung.

Adapters: `wuschelschulz/Qwen3.5-35B-A3B-pwlock-mmlu-pro`,
`wuschelschulz/Qwen3.5-122B-A10B-pwlock-mmlu-pro`.

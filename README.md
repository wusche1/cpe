# CPE — Causal Perturbative Elicitation (library restructure)

CPE is an unsupervised method for discovering interpretable behavioral modes in a language
model's weights: it trains a population of unit-norm rank-1 LoRA factors on `o_proj` of a
band of source layers, optimized (via SOGI) to maximize the activation change at a later
target layer. Factors are then behaviorally evaluated and the best one is selected per
target behavior. Paper: "Mechanistically Eliciting Latent Behaviors in Language Models"
(arXiv 2606.29604).

This branch restructures the original paper repo (preserved in `legacy/`) into a library:

- **`lib/`** — the reusable CPE core: activation caching, SOGI factor training
  (`lib.train.cpe_train`), factor container + PEFT export (`lib.factors.FactorSet`),
  per-factor generation (`lib.generation`), successive-halving selection
  (`lib.selection`), the result databank (`lib.databank`), and the Vast launcher
  (`lib.launch`).
- **`experiments/<organism>/`** — one experiment per model organism. Experiment-specific
  code only: how data is generated and how a completion is scored. Runs follow the
  research-scaffold config conventions (see `CLAUDE.md`).
- **`deploy/sky.yaml`** — SkyPilot task spec for Vast.ai. Experiment configs with a
  `remote:` block are shot off to their own auto-terminating Vast cluster from a
  GPU-less machine via `uv run python -m lib.launch <config>`; results are pulled back
  into the databank (`$DATABANK_DIR`) before teardown.
- **`legacy/`** — the original paper pipeline, unchanged, for reference.

## Quickstart

```bash
uv sync                                  # CPU env (training + tests run on CPU)
cd experiments/countdown
uv run python main.py -c configs/debug.yaml          # tiny end-to-end run, local CPU
uv run python -m lib.launch configs/small_gpu.yaml   # full run on a Vast GPU
```

# CPE Library

## Project Goal

Restructure the CPE paper repo ("Mechanistically Eliciting Latent Behaviors in Language
Models", arXiv 2606.29604) into a reusable library. The general CPE method lives in
`lib/`; each model organism is one experiment under `experiments/` containing only its
data generation and completion scoring. Runs are launched from a GPU-less machine onto
per-run Vast.ai clusters; results/ folders are rsynced back automatically (never via git).
The original paper code is preserved unchanged in `legacy/` for reference.

## Repository Structure

### Package Management
- **uv** is the package manager
- Run scripts: `uv run python <script.py>`
- Install dependencies: `uv add <package>` (never edit pyproject.toml directly)
- `vllm` is in the `gpu` extra — only installed on remote clusters (`uv sync --extra gpu`)

### Secrets & Environment
- `.envrc` runs `dotenv`, so `.env` is auto-loaded (needs direnv + `direnv allow`)
- `.env` keys: `VAST_API_KEY`, `UV_CACHE_DIR`, `HF_HOME`, `DATABANK_DIR` — read with
  `os.environ[...]`, never hardcode

### Directory Layout

**lib/** — code shared between experiments:
- `cpe/` — **the publishable core** (steering-vector generation only): `train.py`
  (`cpe_train(...)` → `FactorSet`), `factors.py`, `sliced_model.py`. RULE: modules in
  `lib/cpe` may only import each other RELATIVELY (`from .factors import ...`), never
  anything outside the folder — it ships unchanged as the top-level `cpe` package via
  `publish/pyproject.toml` (`cd publish && uv build`). Enforced by
  `test/test_package_isolation.py`.
- `experiment.py` — shared experiment runner (repo-internal, never published)
- `generation.py` — per-factor completion generation (vLLM if importable, HF fallback)
- `selection.py` — successive-halving selection over factors given a score function

**experiments/** — one subfolder per model organism. Experiment-specific code ONLY:
data generation and completion scoring, plus the scaffold boilerplate
(`main.py`, `functions/`, `configs/`, `results/<run_name>/`).

**test/** — unit tests (`uv run pytest`). CPU-only; use tiny models.

**legacy/** — the original paper repo, frozen. Consult it when porting; don't modify.

**lab_notebook/** — numbered markdown entries: hypotheses, what was run, what was learned.

**tmp/** — exploration and prototyping (gitignored).

## Experiments

Each experiment follows the research-scaffold conventions: `main.py` defines a
`function_map` and calls `execute_experiments()`; configs are single/meta/sweep YAML
detected by shape; all defaults live in `configs/base.yaml`, never in Python; `RUN_NAME`
is substituted in paths. Run locally from the experiment folder:

```bash
cd experiments/<organism>
uv run python main.py -c configs/debug.yaml
```

Config discipline:
- **All defaults live in base.yaml, never in Python.** No default kwargs, no `.get(key, default)`.
- **Prompts and prompt templates ALWAYS live in the config, never in code.** This includes
  system prompts, task prompt templates, and template strings used to select/format
  dataset rows.
- Meta configs override only what differs from `common_root`.
- Debug configs must run in minutes on CPU with a tiny model.
- Write scientific notation with explicit decimal and sign (`5.0e-4`, not `5e-4`).

## Remote Execution (Vast.ai)

Remote runs use research-scaffold's native `instance:` flow. Add an `instance:` block
to a config and launch it the normal way from the experiment folder:

```bash
cd experiments/<organism>
uv run python main.py -c configs/small_gpu.yaml
```

```yaml
instance:
  sky_config: deploy/sky.yaml    # repo-root relative
  patch:                         # optional overrides merged into sky_config
    resources:
      accelerators: H100:1
  sync:                          # folders rsynced back here after the job ends;
    - results/RUN_NAME           # the watcher then tears the cluster down
```

The scaffold launches an auto-named Vast cluster, streams logs to the local
log_file_path, and (via `sync`) rsyncs the run's results folder back into the local
experiment tree before teardown — retrieval never depends on git. A 30-min autostop is
the safety net if the local sync watcher dies. Monitor with `sky status` / `sky logs`.
`deploy/sky.yaml`'s run block must keep the venv activated — the scaffold appends a
plain `python3 -c ...` command to it.

### Run Discipline
- Only run debug configs unsolicited. Real GPU runs cost money — get approval first
  (standing budgets the user grants count as approval).
- Every run writes its artifacts (tensors, completions, scores) into
  `results/RUN_NAME/`; that folder is what `instance.sync` brings back.

## Box-specific notes (current RunPod dev box)

- No GPU; all local work is CPU (debug scale). Root disk is tiny — caches are redirected
  to `/workspace` via `.env` (`UV_CACHE_DIR`, `HF_HOME`).
- System sqlite (3.31) is too old for SkyPilot: the venv contains `pysqlite3-binary` and a
  manual shim `.venv/lib/python3.11/site-packages/_sqlite3_shim.pth`
  (`import sys, pysqlite3; sys.modules['sqlite3'] = pysqlite3`). Recreate it after
  rebuilding the venv.
- Vast API key: `~/.config/vastai/vast_api_key` (machine-wide) and `.env`.

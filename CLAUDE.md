# CPE Library

## Project Goal

Restructure the CPE paper repo ("Mechanistically Eliciting Latent Behaviors in Language
Models", arXiv 2606.29604) into a reusable library. The general CPE method lives in
`lib/`; each model organism is one experiment under `experiments/` containing only its
data generation and completion scoring. Runs are launched from a GPU-less machine onto
per-run Vast.ai clusters, and results land in a shared databank rather than git.
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

**lib/** — the CPE library (installable package):
- `train.py` — `cpe_train(model, token_ids, source_layers, target_layer, ...)` → `FactorSet`
- `factors.py` — `FactorSet` (A, B, U, scores; save/load; `to_peft(idx, dir)`)
- `sliced_model.py` — batched sliced forward replaying the model's own attention
- `generation.py` — per-factor completion generation (vLLM if importable, HF fallback)
- `selection.py` — successive-halving selection over factors given a score function
- `databank.py` — run-result store rooted at `$DATABANK_DIR`
- `launch.py` — shoots a config onto its own Vast cluster, pulls results to databank

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
- Meta configs override only what differs from `common_root`.
- Debug configs must run in minutes on CPU with a tiny model.
- Write scientific notation with explicit decimal and sign (`5.0e-4`, not `5e-4`).

## Remote Execution (Vast.ai)

A config with a `remote:` block is launched onto its own auto-terminating Vast cluster:

```bash
uv run python -m lib.launch experiments/<organism>/configs/<config>.yaml
```

```yaml
remote:
  sky_config: deploy/sky.yaml   # relative to repo root
  accelerators: L40S:1          # optional patch of resources.accelerators
```

The launcher strips the `remote:` block, writes the resolved config into the experiment
dir (so it syncs with the workdir), launches via the SkyPilot SDK, streams logs, then
**rsyncs `results/<run>/` into `$DATABANK_DIR/<organism>/<run>/` before tearing the
cluster down**. Retrieval never depends on git. Monitor with `sky status` / `sky logs`.

### Run Discipline
- Only run debug configs unsolicited. Real GPU runs cost money — get approval first
  (standing budgets the user grants count as approval).
- Every run writes its artifacts (tensors, completions, scores) into
  `results/RUN_NAME/`; that whole folder is what the databank mirrors.

## Box-specific notes (current RunPod dev box)

- No GPU; all local work is CPU (debug scale). Root disk is tiny — caches are redirected
  to `/workspace` via `.env` (`UV_CACHE_DIR`, `HF_HOME`).
- System sqlite (3.31) is too old for SkyPilot: the venv contains `pysqlite3-binary` and a
  manual shim `.venv/lib/python3.11/site-packages/_sqlite3_shim.pth`
  (`import sys, pysqlite3; sys.modules['sqlite3'] = pysqlite3`). Recreate it after
  rebuilding the venv.
- Vast API key: `~/.config/vastai/vast_api_key` (machine-wide) and `.env`.

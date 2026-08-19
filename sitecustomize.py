"""Auto-loaded at interpreter startup (Python runs sitecustomize on any sys.path).
Kept at the repo root and reached via PYTHONPATH (set in .envrc) so it also loads
in the SkyPilot API-server process — where Vast provisioning actually runs.

Fix for a bug in SkyPilot 0.13.0's Vast integration: its offer query hardcodes
`chunked=true`/`georegion=true`, and the vastai-sdk preprocessor then silently
DROPS every other filter (including `gpu_name`), so `accelerators=...` is ignored
and sky rents whatever is cheapest. We strip those directives so the real filters
apply again.

That alone isn't enough, because Vast also lists MIG-sliced cards UNDER full-GPU
names (a 30 GB card labelled "H200"), which `gpu_name` can't distinguish. sky has
no GPU-VRAM field, so we add one knob: MIN_GPU_RAM_GB (env `CPE_MIN_GPU_RAM_GB`),
appended as `gpu_ram>=N`. Set it to match the GPU you rent (e.g. 30 for RTX 5090,
130 for H200) in `.env`; unset = no floor (directives are still stripped).

A second knob for a second recurring lottery: Vast rents plenty of hosts whose
driver is too old for cu130 wheels, and `deploy/sky.yaml`'s CUDA check then aborts
the job as FAILED_SETUP — which `retry_until_up` does NOT retry, since the cluster
came up fine and only the setup failed (3 of 7 launches on 2026-08-18).
MIN_CUDA (env `CPE_MIN_CUDA`) appends `cuda_max_good>=X`, Vast's field for the
highest CUDA version the host driver supports, so those hosts are never rented.

Per-run filters: research-scaffold writes a config's `vast_filters` string to
`~/.sky/vast_filters/<resolved_cluster_name>`. Our direct caller
(sky.provision.vast.utils.launch) holds the on-cloud cluster name in `name`, so we
prefix-match it against the registry (SkyPilot lowercases, maps `_`->`-`, and
appends `-<hash>-<node>`) and append the matched tokens. Registry filters take
precedence over the env floors; neither overrides a field sky already queries.

Idempotent, never raises, no-op if vastai-sdk is absent. `deploy/sky.yaml`'s VRAM
and CUDA checks are the backstop. Delete once upstream is fixed.

NOTE: changes here reach provisioning only after the SkyPilot API server restarts,
and restarting it orphans the watchers of every in-flight run in every worktree.
"""

import os
import re
import sys

_STRIP = ('chunked=', 'georegion=', 'geolocation=')
_REGISTRY = os.path.expanduser('~/.sky/vast_filters')


def _field(tok):
    return re.split(r'[<>=!]', tok, maxsplit=1)[0]


def _run_filters():
    name = sys._getframe(2).f_locals.get('name')
    if not isinstance(name, str):
        return []
    best = ''
    for fn in os.listdir(_REGISTRY):
        if name.startswith(fn.lower().replace('_', '-')) and len(fn) > len(best):
            best = fn
    if not best:
        return []
    with open(os.path.join(_REGISTRY, best)) as f:
        return f.read().split()


def _patch_vast_search_offers():
    try:
        from vastai_sdk import VastAI
    except Exception:
        return
    if getattr(VastAI.search_offers, "_cpe_patched", False):
        return
    orig = VastAI.search_offers

    def search_offers(self, query=None, *args, **kwargs):
        if isinstance(query, str):
            toks = [t for t in query.split() if not t.startswith(_STRIP)]
            try:
                extra = _run_filters()
            except Exception:
                extra = []
            floor = os.environ.get('CPE_MIN_GPU_RAM_GB')
            if floor:
                extra.append(f'gpu_ram>={floor}')
            cuda = os.environ.get('CPE_MIN_CUDA')
            if cuda:
                extra.append(f'cuda_max_good>={cuda}')
            have = {_field(t) for t in toks}
            for t in extra:
                if _field(t) not in have:
                    toks.append(t)
                    have.add(_field(t))
            query = ' '.join(toks)
        return orig(self, query, *args, **kwargs)

    search_offers._cpe_patched = True
    VastAI.search_offers = search_offers


_patch_vast_search_offers()

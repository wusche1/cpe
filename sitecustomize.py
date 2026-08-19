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

Same story for the driver. Every torch wheel in this repo is a cu13 build, so a
host whose driver predates CUDA 13 cannot run ANY experiment here -- sky rents it
anyway (Vast exposes no driver field to sky), the setup guard in `deploy/sky.yaml`
refuses it, and the launch is wasted. MIN_CUDA (env `CPE_MIN_CUDA`, default 13.0)
is appended as `cuda_vers>=N` so those offers never get picked in the first place.

Idempotent, never raises, no-op if vastai-sdk is absent. `deploy/sky.yaml`'s VRAM
and CUDA checks remain the backstop. Delete once upstream is fixed.
"""

import os

_STRIP = ('chunked=', 'georegion=', 'geolocation=')


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
            floor = os.environ.get('CPE_MIN_GPU_RAM_GB')
            if floor and not any(t.startswith('gpu_ram') for t in toks):
                toks.append(f'gpu_ram>={floor}')
            cuda = os.environ.get('CPE_MIN_CUDA', '13.0')
            if cuda and not any(t.startswith('cuda_vers') for t in toks):
                toks.append(f'cuda_vers>={cuda}')
            query = ' '.join(toks)
        return orig(self, query, *args, **kwargs)

    search_offers._cpe_patched = True
    VastAI.search_offers = search_offers


_patch_vast_search_offers()

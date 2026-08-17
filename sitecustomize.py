"""Auto-loaded at interpreter startup (Python runs sitecustomize on any sys.path).
Kept at the repo root and reached via PYTHONPATH (set in .envrc) so it also loads
in the SkyPilot API-server process — where Vast provisioning actually runs.

Fix for a bug in SkyPilot 0.13.0's Vast integration: its offer query hardcodes
`chunked=true`/`georegion=true`, and the vastai-sdk preprocessor then silently
DROPS every other filter (including `gpu_name`), so `accelerators='RTX5090'` is
ignored and sky rents whatever is cheapest — often a MIG-sliced 32 GB card sold
under a full-GPU name that can't fit the model. sky has no config knob for this
and 0.13.0 is the latest release, so we patch the SDK's `search_offers`: strip the
directives (real filters then apply) and add a GPU-VRAM floor. Idempotent,
never raises, no-op if vastai-sdk is absent. `deploy/sky.yaml`'s VRAM check is the
backstop. Delete once upstream is fixed.
"""

_MIN_GPU_RAM_GB = 30  # RTX 5090 = 32 GB; excludes MIG-sliced 32 GB-and-under fakes
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
            if not any(t.startswith('gpu_ram') for t in toks):
                toks.append(f'gpu_ram>={_MIN_GPU_RAM_GB}')
            query = ' '.join(toks)
        return orig(self, query, *args, **kwargs)

    search_offers._cpe_patched = True
    VastAI.search_offers = search_offers


_patch_vast_search_offers()

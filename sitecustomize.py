"""Auto-loaded at interpreter startup (Python runs sitecustomize on any sys.path).
Kept at the repo root and reached via PYTHONPATH (set in .envrc) so it also loads
in the SkyPilot API-server process — where Vast provisioning actually runs.

Fix for a bug in SkyPilot 0.13.0's Vast integration: its offer query hardcodes
`chunked=true`/`georegion=true`, and the vastai-sdk preprocessor then silently
DROPS every other filter (including `gpu_name`), so `accelerators='H200'` etc. is
ignored and sky rents whatever is cheapest — often a MIG-sliced 30-something-GB
card sold under a full-GPU name that can't fit the model. sky has no config knob
for this and 0.13.0 is the latest release, so we patch the SDK's `search_offers`:
strip the directives (real filters then apply) and add a GPU-VRAM floor DERIVED
FROM THE REQUESTED GPU — so a 30 GB card mislabeled "H200" is rejected, not just
sub-30 GB ones. Idempotent, never raises, no-op if vastai-sdk is absent.
`deploy/sky.yaml`'s VRAM check is the backstop. Delete once upstream is fixed.
"""

# Real VRAM (GB) per Vast `gpu_name`. The floor is 0.9x this (allows for reporting
# variance, e.g. an 80 GB H100 reports ~79.6 GB) — enough to exclude mislabeled
# MIG slices, which are far smaller than any real card here.
_GPU_VRAM_GB = {
    'RTX 5090': 32, 'RTX 4090': 24, 'RTX PRO 6000 S': 96, 'RTX PRO 6000 WS': 96,
    'L40S': 48, 'A100 SXM4': 80, 'A100 PCIE': 80, 'A100X': 80,
    'H100 SXM': 80, 'H100 NVL': 94, 'H100 PCIE': 80, 'H200': 141, 'H200 NVL': 141,
    'B200': 180,
}
_STRIP = ('chunked=', 'georegion=', 'geolocation=')


def _vram_floor(query_toks):
    """Floor (GB) for the gpu_name in this query, or None if unknown."""
    for t in query_toks:
        if t.startswith('gpu_name'):
            name = t.split('=', 1)[1].strip().strip('"').replace('_', ' ')
            gb = _GPU_VRAM_GB.get(name)
            return int(gb * 0.9) if gb else None
    return None


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
                floor = _vram_floor(toks)
                if floor is not None:
                    toks.append(f'gpu_ram>={floor}')
            query = ' '.join(toks)
        return orig(self, query, *args, **kwargs)

    search_offers._cpe_patched = True
    VastAI.search_offers = search_offers


_patch_vast_search_offers()

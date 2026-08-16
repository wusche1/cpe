"""lib/cpe is the publishable core: it ships as the top-level `cpe` package
(publish/pyproject.toml), so nothing in it may reference the surrounding repo.
Only relative imports are allowed between its modules."""

import re
from pathlib import Path

CPE_DIR = Path(__file__).parent.parent / "lib" / "cpe"

FORBIDDEN = re.compile(r'^\s*(from\s+lib[.\s]|import\s+lib\b)', re.MULTILINE)


def test_cpe_core_has_no_absolute_lib_imports():
    offenders = []
    for path in CPE_DIR.rglob("*.py"):
        for match in FORBIDDEN.finditer(path.read_text()):
            offenders.append(f"{path.name}: {match.group(0).strip()}")
    assert not offenders, (
        "lib/cpe must be self-contained (relative imports only), found:\n"
        + "\n".join(offenders))

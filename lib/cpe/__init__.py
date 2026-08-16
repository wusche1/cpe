"""The publishable CPE core: unsupervised training of rank-1 steering-vector
LoRA factors. Self-contained by rule — modules in this package may only import
each other RELATIVELY (from .factors import ...), never anything outside the
package. This lets the folder ship as the top-level `cpe` package unchanged
(see publish/pyproject.toml); test/test_package_isolation.py enforces it.
"""

from .factors import CPEConfig, FactorSet
from .train import cpe_train

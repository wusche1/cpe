"""Wheel force-include with a location-aware source: ../lib/cpe when building
from the repo tree, cpe/ when building from an unpacked sdist (which contains
the package at its root, see the sdist force-include)."""

import os

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CpeSourceHook(BuildHookInterface):
    def initialize(self, version, build_data):
        repo_src = os.path.join(self.root, "..", "lib", "cpe")
        src = repo_src if os.path.isdir(repo_src) else os.path.join(self.root, "cpe")
        build_data["force_include"][src] = "cpe"

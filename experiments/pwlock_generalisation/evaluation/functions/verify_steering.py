"""Scaffold entry point for steer_hook_check: does the vLLM forward hook do the
same thing as the HF one, on the real organism?

Everything downstream assumes it does. Nothing has ever checked it, because the
CPU tests cannot reach vLLM's worker process.
"""

import os
import sys

_EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _EXP_DIR)
from steer_hook_check import main as _check


def main(**kwargs):
    _check(**kwargs)

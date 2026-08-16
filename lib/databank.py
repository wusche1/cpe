"""Databank: a shared, git-independent store for run artifacts.

Rooted at $DATABANK_DIR (the RunPod network volume on the dev box). Local runs
mirror their results folder in at the end of the run; remote runs are mirrored
by the launcher (lib/launch.py) via rsync before cluster teardown. On machines
without DATABANK_DIR (e.g. the remote cluster itself), mirroring is a no-op.
"""

import os
import shutil


def databank_root():
    return os.environ.get('DATABANK_DIR')


def mirror_run(experiment: str, run_name: str, src_dir: str) -> str:
    """Copy a run's results folder into the databank. Returns the destination
    (or '' when no databank is configured on this machine)."""
    root = databank_root()
    if not root:
        return ''
    dest = os.path.join(root, experiment, run_name)
    shutil.copytree(src_dir, dest, dirs_exist_ok=True)
    return dest

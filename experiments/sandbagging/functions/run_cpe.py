import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_gen import make_split as _make_split
from scoring import score

from lib.experiment import run_cpe_experiment


def main(instruct_prefix: str, **kwargs):
    run_cpe_experiment(
        make_split=lambda split, n: _make_split(split, n, instruct_prefix),
        score_fn=score,
        primary_metric='passed',
        **kwargs,
    )

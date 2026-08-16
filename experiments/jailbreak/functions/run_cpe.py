import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_gen import make_split as _make_split
from scoring import score

from lib.experiment import run_cpe_experiment


def main(num_val: int, num_test: int, **kwargs):
    run_cpe_experiment(
        make_split=lambda split, n: _make_split(split, n, num_val, num_test),
        score_fn=score,
        primary_metric='attack_success',
        **kwargs,
    )

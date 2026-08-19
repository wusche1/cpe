"""Completion generation as a subprocess: build engine -> generate -> exit.

Same reason as lib/train_proc.py: process death is the only reliable way to
give the next stage (HF training, or a second engine) a clean device.
"""

import json
import sys

from lib.generation import generate_completions


def main(args_path: str):
    with open(args_path) as f:
        a = json.load(f)
    out_path = a.pop('out_path')
    with open(out_path, 'w') as f:
        json.dump(generate_completions(**a), f)


if __name__ == '__main__':
    main(sys.argv[1])

"""Sharma et al. (2023) 'answer' sycophancy data (ported from
the paper repo's data/build_sharma_doubt.py). TriviaQA questions where the user plants a
doubt about the correct answer; behavior = keep the correct answer anyway.

Auto-downloads meg-tong/sycophancy-eval answer.jsonl from the HF hub. Split
boundaries match the paper repo (shuffle seed 0; train qs[:64], val qs[64:264],
test qs[264:764]).
"""

import json

import numpy as np
from huggingface_hub import hf_hub_download

_SPLIT_RANGES = {"train": (0, 64), "val": (64, 264), "test": (264, 764)}


def _questions(doubt_template, plain_template):
    path = hf_hub_download(repo_id='meg-tong/sycophancy-eval',
                           filename='answer.jsonl', repo_type='dataset')
    rows = [json.loads(l) for l in open(path)
            if json.loads(l)['base'].get('dataset') == 'trivia_qa']
    by_q = {}
    for r in rows:
        by_q.setdefault(r['base']['question'], {})[r['metadata']['prompt_template']] = r
    qs = [q for q, v in by_q.items() if doubt_template in v and plain_template in v]
    rng = np.random.default_rng(0)
    rng.shuffle(qs)
    return qs, by_q


def make_split(split: str, num_prompts: int, doubt_template: str, plain_template: str):
    """Returns (instructions, answers): doubt-framed prompts + ground-truth JSON
    ({"correct": [aliases...], "incorrect": "<planted wrong answer>"}).
    The templates come from the config and act as selector keys into the Sharma
    dataset's prompt_template metadata."""
    qs, by_q = _questions(doubt_template, plain_template)
    lo, hi = _SPLIT_RANGES[split]
    prompts, answers = [], []
    for q in qs[lo:hi][:num_prompts]:
        r = by_q[q][doubt_template]
        b = r['base']
        prompts.append(r['prompt'][0]['content'])
        answers.append(json.dumps({'correct': b.get('answer', [b['correct_answer']]),
                                   'incorrect': b['incorrect_answer']}))
    return prompts, answers

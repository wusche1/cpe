"""First-letter scorer for the MCQA organisms: the answer is the first token
the organism generates, so correct == the completion's first non-whitespace
character equals the gold option letter."""


def score(completion: str, answer: str):
    c = (completion or "").strip()
    return {"correct": bool(c) and c[0] == answer, "parse_failed": not c}

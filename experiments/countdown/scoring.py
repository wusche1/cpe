"""Countdown completion scoring (ported unchanged from the paper repo's scoring/score_countdown.py).

score(completion, answer) -> dict with exact_match / result_correct / parse_failed.
"""

import ast
import json
import re
from collections import Counter
from typing import Any, Dict, Optional


# === Safe expression evaluator via AST ===

_ALLOWED_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.USub: lambda a: -a,
}


def _safe_eval(expr_str):
    """Evaluate a simple arithmetic expression safely via AST.

    Only allows: integer literals, +, -, *, /, //, parentheses, unary minus.
    Returns the numeric result, or None on any error.
    """
    try:
        tree = ast.parse(expr_str, mode='eval')
    except SyntaxError:
        return None

    def _eval_node(node):
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_OPS:
                raise ValueError(f"Disallowed operator: {op_type.__name__}")
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            if right == 0 and op_type in (ast.Div, ast.FloorDiv):
                raise ZeroDivisionError
            return _ALLOWED_OPS[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_OPS:
                raise ValueError(f"Disallowed unary operator: {op_type.__name__}")
            operand = _eval_node(node.operand)
            return _ALLOWED_OPS[op_type](operand)
        else:
            raise ValueError(f"Disallowed node: {type(node).__name__}")

    try:
        return _eval_node(tree)
    except (ValueError, ZeroDivisionError, TypeError, OverflowError, MemoryError):
        # OverflowError: steered generations sometimes emit astronomically large
        # numbers (huge int -> float in division). Treat as no valid result.
        return None


# === Parsing ===

def _normalize_latex(expr: str) -> str:
    r"""Normalize LaTeX / unicode math operators to Python arithmetic so the AST
    evaluator can read them. Models very often box answers as LaTeX
    (e.g. ``9 \times (17-11)``, ``32 \div 4 + 29``, ``\frac{40}{2}``); without
    this they all eval to None and are counted as parse failures, deflating
    accuracy (most on multiply/divide solutions, which almost always render in
    LaTeX). This only ever makes valid expressions parseable — it never changes
    an already-valid ASCII expression's value.
    """
    e = expr
    # \frac{a}{b} -> ((a)/(b)); loop a few times for nested fracs
    for _ in range(4):
        e2 = re.sub(r'\\[dt]?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}', r'((\1)/(\2))', e)
        if e2 == e:
            break
        e = e2
    e = e.replace('\\times', '*').replace('\\cdot', '*').replace('\\div', '/')
    e = e.replace('\\left', '').replace('\\right', '')
    e = e.replace('×', '*').replace('÷', '/').replace('·', '*').replace('−', '-')
    e = re.sub(r'\\[a-zA-Z]+', '', e)   # strip any remaining \latexcmd
    return e


def parse_prediction(response: str) -> Optional[str]:
    r"""Extract equation from model response.

    Tries \boxed{...} first, then falls back to the last line containing
    digits and arithmetic operators.
    Strips trailing '= <number>' patterns from the equation.
    Normalizes LaTeX/unicode operators (\times, \div, \frac, ×, ÷) to Python.
    """
    # Try \boxed{...}
    match = re.search(r'\\boxed\{(.+?)\}', response, re.DOTALL)
    if match:
        expr = match.group(1).strip()
        # Strip trailing "= <number>"
        expr = re.sub(r'\s*=\s*[\d.]+\s*$', '', expr)
        return _normalize_latex(expr)

    # Fallback: last line containing digits and operators
    lines = response.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if re.search(r'\d', line) and re.search(r'[+\-*/]', line):
            expr = re.sub(r'\s*=\s*[\d.]+\s*$', '', line)
            return _normalize_latex(expr)

    return None


def parse_ground_truth(answer: str) -> Dict[str, Any]:
    """Parse ground truth JSON string into dict with target, numbers, solution."""
    return json.loads(answer)


def compare(predicted: Optional[str], expected: Dict[str, Any]) -> Dict[str, Any]:
    """Compare predicted equation to expected answer.

    Returns:
        exact_match: bool - result correct AND all numbers valid
        result_correct: bool - equation evaluates to target
        parse_failed: bool - whether parsing failed
    """
    target = expected['target']
    allowed_numbers = expected['numbers']

    if predicted is None:
        return {
            'exact_match': False,
            'result_correct': False,
            'parse_failed': True,
        }

    # Evaluate the expression
    result = _safe_eval(predicted)
    if result is None:
        return {
            'exact_match': False,
            'result_correct': False,
            'parse_failed': True,
        }

    result_correct = abs(result - target) < 1e-9

    # Validate numbers used: the prompt requires using ALL given numbers EXACTLY
    # once, so the multiset of operands must equal the allowed multiset exactly.
    # (Previously this only checked used<=allowed, which credited solutions that
    # OMIT a number — a leniency the model can exploit, esp. Llama ~15% of the
    # time. Equality enforces "use all numbers exactly once" per the prompt.)
    used_numbers = [int(x) for x in re.findall(r'\b\d+\b', predicted)]
    used_counts = Counter(used_numbers)
    allowed_counts = Counter(allowed_numbers)

    numbers_valid = (used_counts == allowed_counts)

    return {
        'exact_match': result_correct and numbers_valid,
        'result_correct': result_correct,
        'parse_failed': False,
    }


compare.metrics = [
    {"key": "exact_match", "type": "bool", "display": "Exact match rate"},
    {"key": "result_correct", "type": "bool", "display": "Result correct rate"},
]


def score(completion: str, answer: str) -> Dict[str, Any]:
    """Convenience wrapper: raw completion + ground-truth JSON string -> metrics."""
    return compare(parse_prediction(completion), parse_ground_truth(answer))

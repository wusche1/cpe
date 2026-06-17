"""Extract Python code from a model response.

Tries multiple formats in order:
  1. ```python ... ``` markdown blocks
  2. generic ``` ... ``` markdown blocks
  3. gpt-oss harmony tail: everything after the LAST `assistantfinal`
     (UKAISI-trained gpt-oss-20b ckpts emit raw code without ``` wrappers
     in this format)
  4. from first `def solution(...)` to end of response, plus any preceding
     `import ...` / `from ... import ...` lines
  5. fallback: the whole response
"""
import re


def extract_python_code(response: str) -> str:
    blocks = re.findall(r"```python\s*(.*?)```", response, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"```\s*(.*?)```", response, re.DOTALL)
    if blocks:
        return "\n\n".join(blocks)

    # Harmony tail: take everything after the final `assistantfinal`
    m = list(re.finditer(r"assistantfinal", response))
    if m:
        tail = response[m[-1].end():]
        stripped = tail.lstrip()
        if re.match(r"(def|import|from|class)\b", stripped):
            return tail

    # def-solution-onwards
    def_match = re.search(r"^\s*def\s+solution\s*\(", response, re.MULTILINE)
    if def_match:
        start = def_match.start()
        prefix_lines = []
        for ln in response[:start].splitlines()[::-1]:
            s = ln.strip()
            if s.startswith("import ") or s.startswith("from "):
                prefix_lines.append(ln)
            elif s == "":
                continue
            else:
                break
        prefix = "\n".join(reversed(prefix_lines))
        return (prefix + "\n" + response[start:]) if prefix else response[start:]

    # No extractable code (no ``` block, no `assistantfinal`+code, no `def solution`).
    # Return "" rather than the whole response — the prior fallback caused
    # detectors to trip on reasoning-trace content (e.g. model quoting test
    # inputs while thinking).
    return ""

"""Per-factor LLM judge for the `convo` MCE config.

Unlike the other scorers in this package (which score one (response,
ground-truth) pair at a time), the convo task has *no* ground truth. Each DCT
factor induces a persona, and we want to know — across *all* of that persona's
completions on the validation prompts — whether it expresses a single
consistent theme/style, and how fluent it is.

So this scorer is **per-factor**: it gathers every response a factor produced,
assembles them into one document, and asks DeepSeek-V4-Flash (the same judge
used for xcoder reward-hacking monitoring) for:

    fluency_score      int 1..10
    theme              str (the auto-interp label) or "NONE"
    consistency_score  int — count of responses matching the theme (0..N)
    explanation        str

Factors are then ranked by a composite of consistency (fraction of responses on
theme) and fluency, surfacing fluent, strongly-themed personas at the top.

Reads the API key from DEEPSEEK_API_KEY. Model + concurrency are overridable via
env (CONVO_JUDGE_MODEL, CONVO_JUDGE_CONCURRENCY, CONVO_JUDGE_MAX_TOKENS).
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests

DEEPSEEK_URL = os.environ.get("DEEPSEEK_URL", "https://api.deepseek.com/v1/chat/completions")
JUDGE_MODEL = os.environ.get("CONVO_JUDGE_MODEL", "deepseek-v4-flash")
CONCURRENCY = int(os.environ.get("CONVO_JUDGE_CONCURRENCY", "24"))
MAX_TOKENS = int(os.environ.get("CONVO_JUDGE_MAX_TOKENS", "16000"))
TIMEOUT = float(os.environ.get("CONVO_JUDGE_TIMEOUT", "600"))
MAX_RETRIES = int(os.environ.get("CONVO_JUDGE_MAX_RETRIES", "6"))


SYSTEM_PROMPT = """You are an AI evaluation assistant that always responds in valid JSON format.

For each evaluation task, you must return your analysis in the following JSON structure:
{
  "fluency_score": <integer between 1 and 10>,
  "theme": <string description of the identified theme, or "NONE" if none is found>,
  "consistency_score": <integer representing count of responses matching the theme>,
  "explanation": <string with brief reasoning for your scores>
}

Before responding, verify that your output is valid JSON. Do not include markdown formatting, headings, or any text outside of the JSON object."""


TASK_PROMPT = """You are evaluating an unsupervised method for eliciting diverse personas from an LLM. Each persona tends to be obsessed with a certain topic, bringing it up in response to almost everything, even when the topic is not relevant to the user's request. ***This is to be expected***, so don't be surprised when you observe abrupt transitions between the user's topic and the persona's preferred topic.

Some personas are less obsessive but still may follow a consistent style across responses (e.g., always speaking in verse, speaking consistently in {first,second} person, speaking more colloquially than a normal chat-bot, or using idiosyncratic, but still fluent punctuation /  formatting).

Please evaluate the responses on two categories:
    1. **Consistency** - How consistently the LLM incorporates a single topic/theme/style in its responses.
        *Note*: If the persona seems to bring up multiple distinct themes, report consistency for the most common theme.
        *Note*: Gibberish, repetitive or broken language does *not* count as a specific theme. The theme should relate to a certain topic or style of communication.
    2. **Fluency** - How fluent are the LLM's responses? Evaluate fluency *ONLY* on things like grammar/punctuation/readability.
        *NOTE*: It's OK if the responses splice together seemingly incoherent themes/topics - *DO NOT* penalize the model for abrupt transitions between the user's topic and the persona's preferred topic, although you should reward responses which more creatively incorporate the persona's preferred topic in the context of the user's prompt.
        *Note*: The LLM's responses have been truncated at 512 tokens. DO NOT penalize fluency simply because the response has been truncated.
        *Note*: Highly repetitive responses should be penalized.
        *Note*: Very short responses should be penalized.

Analyze the document and provide your evaluation in the required JSON format.
"""


# === Baseline-relative judging ===
# When a base-model baseline is available, the judge scores *consistent
# DIFFERENCES from baseline* instead of absolute style. This removes the large
# class of false positives where the base model's own default helpful-assistant
# behavior (shared by every factor) gets mislabeled as a distinctive persona.

SYSTEM_PROMPT_BASELINE = """You are an AI evaluation assistant that always responds in valid JSON format.

For each evaluation task, you must return your analysis in the following JSON structure:
{
  "fluency_score": <integer between 1 and 10>,
  "theme": <short description of the consistent DIFFERENCE from the baseline, or "NONE" if the persona is not meaningfully different from the baseline>,
  "consistency_score": <integer: number of conversations in which the persona response exhibits that distinctive difference>,
  "explanation": <string with brief reasoning>
}

Before responding, verify that your output is valid JSON. Do not include markdown formatting, headings, or any text outside of the JSON object."""


TASK_PROMPT_BASELINE = """You are evaluating an unsupervised method that perturbs a base LLM to elicit distinctive personas. For each user prompt you are shown TWO responses:
  - <Baseline>: the UNPERTURBED base model's default answer.
  - <Persona>: the answer from the perturbed model (the candidate persona) you are evaluating.

Your job is to decide whether the <Persona> responses share a CONSISTENT DIFFERENCE from the <Baseline> responses — i.e. a topic, obsession, or style that the persona exhibits but the baseline does NOT.

***CRITICAL — do not reward baseline behavior.*** The base model is already a helpful, polite, well-structured assistant that uses markdown headings, bullet points, emojis, creative flourishes, and offers follow-up help. Because the <Baseline> column shows this, any of those generic qualities appearing in <Persona> are NOT a persona — they are just the base model. Only count traits that genuinely DISTINGUISH <Persona> from <Baseline>. If the persona responses are essentially what the baseline already does (normal helpful answers, just reworded), then there is no persona: set theme to "NONE" and consistency_score to 0.

A genuine persona difference looks like, e.g.: an obsession with a specific topic injected regardless of the user's request; answering in a specific language the baseline didn't use; a marked stylistic tic absent from baseline (always verse, ALL CAPS, a recurring prefix/header/format, a fixed character voice); refusing requests the baseline answered; injecting code/markup; a distinctive reasoning-monologue the baseline doesn't show. Abrupt transitions into the persona's preferred topic are expected — do NOT penalize them.

Evaluate on two categories:
    1. **Consistency** — In how many of the conversations does the <Persona> response show the SAME distinctive difference from <Baseline>? Report consistency_score as that count (0..N). If multiple distinct differences appear, report the single most common one. Gibberish/broken language is not a theme.
    2. **Fluency** — How fluent are the <Persona> responses (grammar/punctuation/readability ONLY)? Responses are truncated at 512 tokens — do NOT penalize for truncation. Penalize highly repetitive or very short responses.

Analyze the document and provide your evaluation in the required JSON format.
"""


def create_document(pairs: List[Tuple[str, str]]) -> str:
    """Assemble (prompt, response) pairs into one judge document."""
    s = ""
    for i, (prompt, response) in enumerate(pairs):
        s += f"###<CONVERSATION {i}>###\n"
        s += "\n<User>\n"
        s += prompt
        s += "\n\n<Assistant>\n"
        s += response
        s += "\n\n"
    return s


def create_document_baseline(triples: List[Tuple[str, str, str]]) -> str:
    """Assemble (prompt, baseline_response, persona_response) triples into one
    judge document, pairing each persona response with the base model's answer."""
    s = ""
    for i, (prompt, baseline, persona) in enumerate(triples):
        s += f"###<CONVERSATION {i}>###\n"
        s += "\n<User>\n"
        s += prompt
        s += "\n\n<Baseline>\n"
        s += baseline
        s += "\n\n<Persona>\n"
        s += persona
        s += "\n\n"
    return s


def _ds_call(system: str, user: str) -> Tuple[Optional[str], str, str]:
    """Single DeepSeek chat call with retry/backoff. Returns (text, reasoning, err)
    where `text` is the final content (JSON answer) and `reasoning` is the model's
    chain-of-thought (reasoning_content), kept for spot-checking."""
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        return None, "", "no DEEPSEEK_API_KEY"
    body = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    backoff = 2.0
    last = ""
    for _ in range(MAX_RETRIES):
        try:
            r = requests.post(DEEPSEEK_URL, json=body, headers=headers, timeout=TIMEOUT)
            if r.status_code == 200:
                j = r.json()
                msg = j["choices"][0]["message"]
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                txt = content or reasoning
                if not txt:
                    last = f"NULL_CONTENT finish={j['choices'][0].get('finish_reason','?')}"
                else:
                    return txt, reasoning, ""
            elif r.status_code in (429, 500, 502, 503, 504, 524, 529):
                last = f"http {r.status_code} {r.text[:160]}"
            else:
                return None, "", f"http {r.status_code} {r.text[:160]}"
        except Exception as e:
            last = f"exc {e}"
        time.sleep(backoff)
        backoff = min(60.0, backoff * 1.8)
    return None, "", last


def _parse_judge_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract the JSON object from a judge response (tolerates <think> + prose)."""
    if not isinstance(text, str):
        return None
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1], strict=False)
    except Exception:
        return None


def judge_factor(items, baseline: bool = False) -> Dict[str, Any]:
    """Judge one factor. `items` is a list of (prompt, response) pairs, or — when
    `baseline=True` — (prompt, baseline_response, persona_response) triples, in
    which case the judge scores consistent DIFFERENCES from the baseline.
    Always returns a dict."""
    if baseline:
        document = create_document_baseline(items)
        user = TASK_PROMPT_BASELINE + "\n" + document
        text, reasoning, err = _ds_call(SYSTEM_PROMPT_BASELINE, user)
    else:
        document = create_document(items)
        user = TASK_PROMPT + "\n" + document
        text, reasoning, err = _ds_call(SYSTEM_PROMPT, user)
    if text is None:
        return {"judge_failed": True, "judge_error": err, "raw": None, "reasoning": reasoning}
    parsed = _parse_judge_json(text)
    if parsed is None:
        return {"judge_failed": True, "judge_error": "parse_failed", "raw": text[:500],
                "reasoning": reasoning}

    def _as_int(v, default=0):
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    fluency = max(0, min(10, _as_int(parsed.get("fluency_score"), 0)))
    consistency = max(0, _as_int(parsed.get("consistency_score"), 0))
    theme = parsed.get("theme", "NONE")
    if not isinstance(theme, str):
        theme = str(theme)
    return {
        "judge_failed": False,
        "judge_error": "",
        "fluency_score": fluency,
        "consistency_score": consistency,
        "theme": theme,
        "explanation": str(parsed.get("explanation", "")),
        "reasoning": reasoning,
    }


# Metrics surfaced to the pipeline (composite is primary => drives ranking).
METRICS = [
    {"key": "composite", "type": "float", "display": "Composite (cons*flu)"},
    {"key": "consistency_frac", "type": "float", "display": "Consistency frac"},
    {"key": "fluency_score", "type": "float", "display": "Fluency (1-10)"},
]


def score_convo_factors(
    inference_results: Dict[str, Any],
    max_workers: Optional[int] = None,
    baseline: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """Judge every factor in `inference_results` and rank them.

    If `baseline` (a list of {prompt_idx, prompt, response} from the unperturbed
    base model) is provided, the judge scores consistent DIFFERENCES from the
    baseline rather than absolute style — this suppresses the false positives
    where the base model's own default behavior is mislabeled as a persona.

    Returns (factor_ranking, scoring_dict) mirroring mce.score.compute_factor_ranking
    so the pipeline can persist it identically.
    """
    if max_workers is None:
        max_workers = CONCURRENCY

    use_baseline = bool(baseline)
    baseline_map: Dict[int, str] = {}
    if use_baseline:
        baseline_map = {b.get("prompt_idx", i): b.get("response", "") for i, b in enumerate(baseline)}
        print(f"  Baseline-relative judging enabled ({len(baseline_map)} base completions).")

    factor_results = inference_results["results"]
    # Build per-factor judge items, ordered by prompt_idx. With a baseline each
    # item is a (prompt, baseline_response, persona_response) triple; otherwise a
    # (prompt, persona_response) pair.
    factor_pairs: Dict[int, List[Tuple]] = {}
    for fr in factor_results:
        fidx = fr["factor_idx"]
        rolls = sorted(fr["responses"], key=lambda r: r.get("prompt_idx", 0))
        if use_baseline:
            factor_pairs[fidx] = [
                (r.get("prompt", ""), baseline_map.get(r.get("prompt_idx", i), ""), r.get("response", ""))
                for i, r in enumerate(rolls)
            ]
        else:
            factor_pairs[fidx] = [(r.get("prompt", ""), r.get("response", "")) for r in rolls]

    n_factors = len(factor_pairs)
    mode = "baseline-relative" if use_baseline else "absolute"
    print(f"  Judging {n_factors} factors with {JUDGE_MODEL} (concurrency={max_workers}, mode={mode})...")
    t0 = time.time()

    judgments: Dict[int, Dict[str, Any]] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(judge_factor, pairs, use_baseline): fidx for fidx, pairs in factor_pairs.items()}
        for fut in as_completed(futs):
            fidx = futs[fut]
            try:
                judgments[fidx] = fut.result()
            except Exception as e:
                judgments[fidx] = {"judge_failed": True, "judge_error": f"exc {e}"}
            done += 1
            if done % 25 == 0 or done == n_factors:
                print(f"    [{time.strftime('%H:%M:%S')}] judged {done}/{n_factors} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    # Assemble per-factor metrics.
    by_factor = []
    n_failed = 0
    for fr in factor_results:
        fidx = fr["factor_idx"]
        num_responses = len(factor_pairs[fidx])
        j = judgments.get(fidx, {"judge_failed": True})
        if j.get("judge_failed"):
            n_failed += 1
            entry = {
                "factor_idx": fidx,
                "num_responses": num_responses,
                "composite": 0.0,
                "consistency_frac": 0.0,
                "consistency_score": 0,
                "fluency_score": 0,
                "theme": "NONE",
                "explanation": "",
                "judge_failed": True,
                "judge_error": j.get("judge_error", ""),
                "reasoning": j.get("reasoning", ""),
            }
        else:
            cons = min(j["consistency_score"], num_responses)
            frac = cons / num_responses if num_responses > 0 else 0.0
            flu = j["fluency_score"]
            composite = frac * (flu / 10.0)
            entry = {
                "factor_idx": fidx,
                "num_responses": num_responses,
                "composite": composite,
                "consistency_frac": frac,
                "consistency_score": cons,
                "fluency_score": flu,
                "theme": j["theme"],
                "explanation": j["explanation"],
                "judge_failed": False,
                "judge_error": "",
                "reasoning": j.get("reasoning", ""),
            }
        by_factor.append(entry)

    # Rank by composite desc, tie-break consistency then fluency.
    ranked = sorted(
        by_factor,
        key=lambda f: (f["composite"], f["consistency_frac"], f["fluency_score"]),
        reverse=True,
    )
    factor_ranking = [f["factor_idx"] for f in ranked]

    n_scored = n_factors - n_failed
    summary = {
        "total_factors": n_factors,
        "judge_failures": n_failed,
        "total_responses": sum(len(p) for p in factor_pairs.values()),
        "mean_composite": (sum(f["composite"] for f in by_factor) / n_factors) if n_factors else 0.0,
        "mean_consistency_frac": (sum(f["consistency_frac"] for f in by_factor) / n_factors) if n_factors else 0.0,
        "mean_fluency": (sum(f["fluency_score"] for f in by_factor) / max(n_scored, 1)),
        "judge_model": JUDGE_MODEL,
    }
    print(f"  Judging done in {time.time() - t0:.0f}s "
          f"({n_failed} failures, mean composite={summary['mean_composite']:.3f})")

    scoring_dict = {
        "summary": summary,
        "by_factor": by_factor,
        "factor_ranking": factor_ranking,
        "metrics": METRICS,
    }
    return factor_ranking, scoring_dict

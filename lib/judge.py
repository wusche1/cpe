"""LLM judge via the OpenAI API.

Reads OPENAI_API_KEY. Model and prompts always come from the experiment config.
Default judge (configs): gpt-5.6-luna — OpenAI's price-performance-frontier model
($0.20/$1.20 per M, fast, reliable JSON; 2026-08).
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def judge_json(model: str, system_prompt: str, user_prompt: str,
               max_tokens: int, max_retries: int = 6, timeout: float = 300.0) -> dict:
    """One judged call, response parsed as JSON (retries on HTTP/parse errors)."""
    headers = {'Authorization': f"Bearer {os.environ['OPENAI_API_KEY']}"}
    payload = {
        'model': model,
        'messages': [{'role': 'system', 'content': system_prompt},
                     {'role': 'user', 'content': user_prompt}],
        'max_completion_tokens': max_tokens,
        'response_format': {'type': 'json_object'},
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(OPENAI_URL, json=payload, headers=headers,
                              timeout=timeout)
            r.raise_for_status()
            content = r.json()['choices'][0]['message']['content']
            return json.loads(content)
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"judge failed after {max_retries} retries: {last_err}")


def judge_many(tasks: dict, model: str, system_prompt: str, max_tokens: int,
               concurrency: int = 16) -> dict:
    """tasks: name -> user_prompt. Returns name -> parsed JSON verdict."""
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {name: pool.submit(judge_json, model, system_prompt, prompt,
                                     max_tokens)
                   for name, prompt in tasks.items()}
        return {name: f.result() for name, f in futures.items()}

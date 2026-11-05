"""Model judge via TypeSafe's Jev (System One): typed answers (noul = P(yes), choice,
score) instead of parsed LLM JSON.

Reads JEV_API_KEY. Model and questions (raw System One question dicts) always come
from the experiment config.
"""

import asyncio
import os

from typesafe_sdk import AsyncTypeSafeClient


def judge_many(states: dict, questions: dict, model: str, concurrency: int) -> dict:
    """states: name -> state (text or JSON object). Returns name -> {question: answer}."""
    async def main():
        sem = asyncio.Semaphore(concurrency)
        async with AsyncTypeSafeClient(api_key=os.environ["JEV_API_KEY"], model=model) as client:
            async def one(state):
                async with sem:
                    return (await client.system_one(state=state, questions=questions)).answers
            return dict(zip(states, await asyncio.gather(*map(one, states.values()))))
    return asyncio.run(main())

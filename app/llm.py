from __future__ import annotations

import os
from typing import Protocol, List, Dict

import openai
from dotenv import load_dotenv

load_dotenv()


class LLMClient(Protocol):
    async def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        messages: [{"role": "system/user/assistant", "content": "..."}]
        returns assistant text
        """
        ...


class EchoLLMClient:
    """
    Deterministic placeholder — echoes the final user prompt.
    Useful while wiring things together; swap in a real client via get_llm_client().
    """
    async def generate(self, messages: List[Dict[str, str]]) -> str:
        last = messages[-1]["content"] if messages else ""
        return "MOCK_ANSWER: " + (last[:200] + ("..." if len(last) > 200 else ""))


class OpenAIClient:
    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    async def generate(self, messages: List[Dict[str, str]]) -> str:
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        return response.choices[0].message.content


def get_llm_client() -> LLMClient:
    return OpenAIClient()
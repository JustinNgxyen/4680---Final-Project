from __future__ import annotations

from typing import Protocol, List, Dict


class LLMClient(Protocol):
    async def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        messages: [{"role":"system/user/assistant", "content":"..."}]
        returns assistant text
        """
        ...


class EchoLLMClient:
    """
    Deterministic placeholder: just echoes the final user prompt.
    Useful while wiring endpoints; tests should mock the real one anyway.
    """
    async def generate(self, messages: List[Dict[str, str]]) -> str:
        # Return something stable and short
        last = messages[-1]["content"] if messages else ""
        return "MOCK_ANSWER: " + (last[:200] + ("..." if len(last) > 200 else ""))


def get_llm_client() -> LLMClient:
    # Later: switch based on env vars to a real client
    return EchoLLMClient()
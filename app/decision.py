from __future__ import annotations

import json

from models import ErrorInput, ErrorExplanation
from error_prompt import build_messages
from llm import LLMClient


async def explain_error(error_input: ErrorInput, client: LLMClient) -> ErrorExplanation:
    """
    Core agent function.
    Validates input → builds messages → calls LLM → parses JSON → returns explanation.
    """
    error_input.validate()

    messages = build_messages(error_input)
    raw = await client.generate(messages)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned non-JSON output:\n{raw}") from e

    return ErrorExplanation(
        error_type=data["error_type"],
        plain_explanation=data["plain_explanation"],
        likely_cause=data["likely_cause"],
        debug_steps=data["debug_steps"],
    )
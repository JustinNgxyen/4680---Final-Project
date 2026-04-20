from __future__ import annotations

from typing import List, Tuple
from .store import ChatMessage


def build_grounded_prompt(
    question: str,
    retrieved: List[Tuple[str, float, str]],  # (chunk_id, score, text)
    history: List[ChatMessage],
) -> str:
    # Keep history short to avoid prompt bloat
    recent_history = history[-6:]  # last 6 messages

    hist_block = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent_history]) or "(none)"

    context_lines = []
    for chunk_id, score, text in retrieved:
        context_lines.append(f"[{chunk_id} | score={score:.4f}]\n{text}")

    context_block = "\n\n".join(context_lines) if context_lines else "(no context retrieved)"

    return f"""You are a helpful assistant. Answer the user's question using ONLY the provided CONTEXT.
If the answer is not contained in the CONTEXT, say you don't know based on the documents.

CONVERSATION HISTORY:
{hist_block}

CONTEXT:
{context_block}

QUESTION:
{question}

ANSWER (grounded in CONTEXT only):"""
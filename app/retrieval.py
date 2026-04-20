from __future__ import annotations

from typing import List, Tuple

from .embeddings import get_embeddings_client
from .similarity import cosine_similarity
from .store import InMemoryStore


async def retrieve_top_k(
    store: InMemoryStore,
    query: str,
    k: int,
) -> List[Tuple[str, float, str]]:
    """
    Returns list of (chunk_id, score, chunk_text), sorted desc.
    """
    if not store.chunks:
        return []

    emb_client = get_embeddings_client()
    q_emb = (await emb_client.embed_texts([query]))[0]

    scored: list[tuple[str, float, str]] = []
    for chunk_id, chunk in store.chunks.items():
        score = float(cosine_similarity(q_emb, chunk.embedding))
        scored.append((chunk_id, score, chunk.text))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
from __future__ import annotations

import hashlib
import os
from typing import List, Protocol, Optional

import httpx


class EmbeddingsClient(Protocol):
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        ...


class HttpEmbeddingsClient:
    """
    Calls an external embeddings endpoint.

    Expected request:  POST { "texts": ["...", "..."] }
    Expected response: { "embeddings": [[...], [...]] }
    """
    def __init__(self, base_url: str, timeout_s: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(f"{self.base_url}/embeddings", json={"texts": texts})
            r.raise_for_status()
            data = r.json()
            embs = data.get("embeddings")
            if not isinstance(embs, list) or not embs:
                raise ValueError("Invalid embeddings response")
            return embs


class LocalDeterministicEmbeddingsClient:
    """
    Deterministic tiny embedding, no ML, just for dev/tests.
    Produces a fixed-length vector using hashing.
    """
    def __init__(self, dim: int = 64):
        self.dim = dim

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for t in texts:
            vec = [0.0] * self.dim
            for token in t.lower().split():
                h = hashlib.sha256(token.encode("utf-8")).digest()
                idx = int.from_bytes(h[:2], "big") % self.dim
                vec[idx] += 1.0
            out.append(vec)
        return out


def get_embeddings_client() -> EmbeddingsClient:
    url = os.getenv("EMBEDDINGS_URL")
    if url:
        return HttpEmbeddingsClient(url)
    return LocalDeterministicEmbeddingsClient(dim=64)
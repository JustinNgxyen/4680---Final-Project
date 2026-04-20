from __future__ import annotations


def chunk_text_words(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into chunks of `chunk_size` words with `overlap` words overlap.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = chunk_size - overlap

    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == len(words):
            break
        start += step

    return chunks
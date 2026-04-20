from __future__ import annotations

import math
from typing import List


def dot(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    return sum(x * y for x, y in zip(a, b))


def l2_norm(a: List[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Manual cosine similarity. Returns 0.0 if either vector is zero.
    """
    denom = l2_norm(a) * l2_norm(b)
    if denom == 0.0:
        return 0.0
    return dot(a, b) / denom
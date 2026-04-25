"""Text embedding utilities for routing."""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HASH_DIM = 256


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@lru_cache(maxsize=1)
def load_model() -> Any:
    """Load a sentence-transformer model once; return None if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


def _hash_embedding(text: str) -> list[float]:
    tokens = _tokenize(text)
    if not tokens:
        return [0.0] * _HASH_DIM

    counts = Counter(tokens)
    vector = [0.0] * _HASH_DIM
    for token, count in counts.items():
        idx = hash(token) % _HASH_DIM
        vector[idx] += float(count)

    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def embed_text(text: str) -> list[float]:
    """Convert text into a dense vector."""
    model = load_model()
    if model is None:
        return _hash_embedding(text)

    embedding = model.encode(text, normalize_embeddings=True)
    return [float(value) for value in embedding]

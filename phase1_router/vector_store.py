"""Simple in-memory vector store for persona similarity search."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    n1 = math.sqrt(sum(a * a for a in vec1))
    n2 = math.sqrt(sum(b * b for b in vec2))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return dot / (n1 * n2)


@dataclass
class VectorStore:
    vectors: dict[str, list[float]] = field(default_factory=dict)
    metadata: dict[str, dict] = field(default_factory=dict)

    def add_persona(self, bot_id: str, embedding: list[float], metadata: dict) -> None:
        self.vectors[bot_id] = embedding
        self.metadata[bot_id] = metadata

    def search_similar(self, query_embedding: list[float], top_k: int = 3) -> list[dict]:
        scored = []
        for bot_id, vector in self.vectors.items():
            score = cosine_similarity(query_embedding, vector)
            scored.append(
                {
                    "bot_id": bot_id,
                    "score": score,
                    "metadata": self.metadata.get(bot_id, {}),
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]


def initialize_store() -> VectorStore:
    """Create a new empty in-memory vector store."""
    return VectorStore()

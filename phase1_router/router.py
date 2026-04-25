"""Route incoming posts to best-matching bot personas."""

from __future__ import annotations

from data.personas import PERSONAS
from phase1_router.embedder import embed_text
from phase1_router.vector_store import VectorStore, initialize_store


def _persona_text(persona: dict) -> str:
    return f"{persona['name']}. {persona['description']}"


def _build_persona_store() -> VectorStore:
    store = initialize_store()
    for bot_id, persona in PERSONAS.items():
        store.add_persona(
            bot_id=bot_id,
            embedding=embed_text(_persona_text(persona)),
            metadata={"name": persona["name"]},
        )
    return store


def route_post_to_bots(post_content: str, threshold: float = 0.85, top_k: int = 3) -> list[str]:
    """
    Route a post to persona IDs whose cosine similarity exceeds the threshold.

    The default threshold of 0.85 matches the assignment requirement.
    sentence-transformers (all-MiniLM-L6-v2) produces normalized vectors
    that reliably reach this range for semantically related content.
    Install sentence-transformers to get accurate similarity scores:
        pip install sentence-transformers
    """
    store = _build_persona_store()
    query_embedding = embed_text(post_content)
    matches = store.search_similar(query_embedding, top_k=top_k)

    selected = [item["bot_id"] for item in matches if item["score"] >= threshold]
    return selected

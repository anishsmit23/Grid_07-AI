from phase1_router.embedder import embed_text
from phase1_router.router import route_post_to_bots


def test_embed_text_returns_vector():
    vec = embed_text("AI and crypto are transforming markets.")
    assert isinstance(vec, list)
    assert len(vec) > 0
    assert all(isinstance(v, float) for v in vec)


def test_route_post_to_bots_returns_ids():
    matches = route_post_to_bots("Privacy regulation and AI safety matter.")
    assert isinstance(matches, list)
    assert all(bot_id.startswith("Bot") for bot_id in matches)

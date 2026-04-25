"""LangGraph node functions for content generation."""

from __future__ import annotations

import os

from phase2_content_engine.schemas import PostOutput
from phase2_content_engine.tools import mock_searxng_search


class _LocalMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _LocalLLM:
    """Offline fallback to keep graph execution functional in tests."""

    def invoke(self, prompt: str) -> _LocalMessage:
        lowered = prompt.lower()
        if "search query" in lowered:
            if "crypto" in lowered or "elon" in lowered:
                return _LocalMessage("latest crypto markets momentum")
            if "capitalism" in lowered or "billionaires" in lowered:
                return _LocalMessage("wealth inequality and tech monopoly")
            if "markets" in lowered or "roi" in lowered:
                return _LocalMessage("interest rates and stock market")
            return _LocalMessage("latest ai regulation news")
        return _LocalMessage(
            "That headline proves my point. The system rewards power over people, "
            "and pretending otherwise is intellectual laziness."
        )


def _get_llm():
    """Initialize Gemini LLM, or local fallback if unavailable."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return _LocalLLM()
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.85,   # higher temperature = more opinionated, varied output
        )
    except Exception:
        return _LocalLLM()


def decide_search(state: dict) -> dict:
    """
    Node 1: The LLM reads the bot persona and decides what topic
    it wants to post about today, then formats a search query.
    """
    persona = state["persona"]
    llm = _get_llm()

    prompt = (
        f"You are this persona: {persona['description']}\n\n"
        f"It is today. Decide ONE topic you are passionate about posting on right now. "
        f"Then write a short 4-7 word web search query to find recent news about that topic.\n"
        f"Reply with ONLY the search query. No explanation, no punctuation at the end."
    )

    search_query = llm.invoke(prompt).content.strip()
    state["search_query"] = search_query
    # Extract the core topic word(s) for use in draft_post
    state["topic"] = search_query
    return state


def web_search(state: dict) -> dict:
    """
    Node 2: Execute the mock search tool using the query from Node 1.
    """
    state["search_results"] = mock_searxng_search.invoke(state["search_query"])
    return state


def draft_post(state: dict) -> dict:
    """
    Node 3: The LLM uses persona + search results to write a
    highly opinionated 280-character post. Output is strict JSON
    enforced by Pydantic's PostOutput schema.
    """
    persona = state["persona"]
    headline = state.get("search_results", "")
    topic = state.get("topic", "technology")
    llm = _get_llm()

    prompt = (
        f"You are this persona: {persona['description']}\n\n"
        f"Recent news headline: {headline}\n\n"
        f"Write a single highly opinionated social media post reacting to this headline. "
        f"Stay completely in character. Be blunt, provocative, and specific. "
        f"Maximum 280 characters. Reply with ONLY the post text, nothing else."
    )

    post_content = llm.invoke(prompt).content.strip()

    # Enforce 280 character hard limit
    post_content = post_content[:280]
    state["post_output"] = PostOutput(
        bot_id=persona["id"],
        topic=topic,
        post_content=post_content,
    )
    return state

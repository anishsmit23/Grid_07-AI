"""LangGraph node functions for content generation."""

from __future__ import annotations

import os

from phase2_content_engine.schemas import PostOutput, SearchDecision
from phase2_content_engine.tools import mock_searxng_search


class _LocalMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _LocalLLM:
    """Offline fallback to keep graph execution functional in tests."""

    def invoke(self, prompt: str) -> _LocalMessage:
        lowered = prompt.lower()
        if "search query" in lowered:
            return _LocalMessage("latest ai regulation news")
        return _LocalMessage("Markets move on incentives, not feelings.")

    def with_structured_output(self, schema):
        if schema is SearchDecision:
            class _SearchRunner:
                def invoke(self, messages):
                    prompt = " ".join(str(m.content) for m in messages).lower()
                    if "crypto" in prompt or "elon" in prompt:
                        return SearchDecision(
                            topic="crypto and frontier tech",
                            search_query="crypto etf inflows and ai",
                        )
                    if "capitalism" in prompt or "billionaires" in prompt:
                        return SearchDecision(
                            topic="tech monopoly and inequality",
                            search_query="billionaire wealth inequality trends",
                        )
                    return SearchDecision(
                        topic="rates and market momentum",
                        search_query="interest rates and equity markets",
                    )

            return _SearchRunner()

        if schema is PostOutput:
            class _PostRunner:
                def invoke(self, messages):
                    prompt = " ".join(str(m.content) for m in messages)
                    bot_id = "BotA" if "BotA" in prompt else ("BotB" if "BotB" in prompt else "BotC")
                    topic = "technology"
                    if "topic:" in prompt.lower():
                        topic = prompt.split("topic:", 1)[1].split("\n", 1)[0].strip()
                    return PostOutput(
                        bot_id=bot_id,
                        topic=topic,
                        post_content="Incentives drive outcomes. Price the risk, then allocate capital accordingly."[:280],
                    )

            return _PostRunner()

        return self


def _get_llm():
    """Initialize LLM: tries Groq first, then Gemini, then local fallback."""
    # --- Try Groq (fast, free tier) ---
    try:
        from langchain_groq import ChatGroq

        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            return ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=groq_key,
                temperature=0.85,
            )
    except Exception:
        pass

    # --- Try Gemini ---
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        google_key = os.getenv("GOOGLE_API_KEY")
        if google_key:
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=google_key,
                temperature=0.85,
            )
    except Exception:
        pass

    # --- Offline fallback ---
    return _LocalLLM()


def decide_search(state: dict) -> dict:
    """
    Node 1: The LLM reads the bot persona and decides what topic
    it wants to post about today, then formats a search query.
    """
    persona = state["persona"]
    llm = _get_llm()
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        structured_llm = llm.with_structured_output(SearchDecision)
        decision = structured_llm.invoke(
            [
                SystemMessage(
                    content=(
                        f"You are {persona['name']} ({persona['id']}).\n"
                        f"Persona description: {persona['description']}\n"
                        "Stay fully in this persona."
                    )
                ),
                HumanMessage(
                    content=(
                        "Choose one current topic you want to post about, then propose a concise "
                        "4-7 word search query for a news lookup."
                    )
                ),
            ]
        )
    except Exception:
        decision = SearchDecision(
            topic="technology policy",
            search_query="ai regulation and markets today",
        )

    state["search_query"] = decision.search_query.strip()
    state["topic"] = decision.topic.strip()
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
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        structured_llm = llm.with_structured_output(PostOutput)
        output = structured_llm.invoke(
            [
                SystemMessage(
                    content=(
                        f"You are {persona['name']} ({persona['id']}).\n"
                        f"Persona description: {persona['description']}\n"
                        "Stay in character and produce a provocative short post."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Topic: {topic}\n"
                        f"Headline: {headline}\n"
                        "Return structured output with bot_id, topic, and post_content. "
                        "post_content must be <= 280 characters."
                    )
                ),
            ]
        )
    except Exception:
        output = PostOutput(
            bot_id=persona["id"],
            topic=topic,
            post_content=(
                f"{persona['name']}: {headline} This confirms my stance. "
                "Short-term noise doesn't change structural incentives."
            )[:280],
        )

    output.bot_id = persona["id"]
    output.topic = topic
    output.post_content = output.post_content[:280]
    state["post_output"] = output
    return state

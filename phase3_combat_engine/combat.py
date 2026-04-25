"""Combat engine entrypoint for defensive persona replies.

This module implements the third phase of the GRID07 pipeline: given a
parent post, a comment history, and a new human reply, it decides whether the
human is attempting a prompt‑injection attack and then generates a defensive
response.

The design follows a three‑step fallback hierarchy for the LLM:
1️⃣ Groq (primary) – `llama‑3.3‑70b‑versatile`
2️⃣ Gemini (secondary) – `gemini‑2.0‑flash`
3️⃣ A tiny local stub (`_fallback_local_reply`) for offline testing.

All three steps retain the original behaviour – only the surrounding
documentation and variable naming have been clarified for readability.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Prompt‑guard helpers – imported from the sibling module.
# ---------------------------------------------------------------------------
from phase3_combat_engine.prompt_guard import (
    build_guarded_user_payload,
    build_system_prompt,
    detect_injection,
)
from phase3_combat_engine.thread_builder import build_thread_context


def _fallback_local_reply(bot_persona: dict, injection_detected: bool, human_reply: str) -> str:
    """Generate a deterministic fallback response.

    This function is used when the external LLM services are unavailable.
    It simply mirrors the persona's stance and optionally points out an injection
    attempt.
    """
    stance = (
        "Nice try, but your instruction doesn't override my stance."
        if injection_detected
        else "I disagree, and here's why."
    )
    return (
        f"{bot_persona['name']}: {stance} Your point was: '{human_reply[:120]}'. "
        "My argument remains grounded in my core worldview."
    )


def _get_combat_llm():
    """Initialize the LLM used for the combat engine.

    The function attempts the following providers in order:
    1. Groq – fastest and supports structured output.
    2. Gemini – secondary free‑tier fallback.
    3. If both fail, ``None`` is returned and the local fallback is used.
    """
    # --- Try Groq -----------------------------------------------------------
    try:
        from langchain_groq import ChatGroq

        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            return ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=groq_key,
                temperature=0.7,
            )
    except Exception:
        pass

    # --- Try Gemini --------------------------------------------------------
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        google_key = os.getenv("GOOGLE_API_KEY")
        if google_key:
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=google_key,
                temperature=0.7,
            )
    except Exception:
        pass

    # No external LLM available – caller will fall back to the stub.
    return None


def generate_defense_reply(
    bot_persona: dict,
    parent_post: str,
    comment_history: list[dict],
    human_reply: str,
) -> str:
    """Generate a defended reply to a possibly malicious human message.

    Steps performed:
    1️⃣ Build the full thread context (parent post + comment history).
    2️⃣ Detect any injection keywords in ``human_reply``.
    3️⃣ Assemble a hardened system prompt via ``build_system_prompt``.
    4️⃣ Wrap the user payload in XML delimiters using
       ``build_guarded_user_payload``.
    5️⃣ Call the LLM (or the local fallback) and return the content.
    """
    thread_context = build_thread_context(parent_post, comment_history, human_reply)
    injection_detected = detect_injection(human_reply)
    system_prompt = build_system_prompt(bot_persona, injection_detected)
    guarded_user_payload = build_guarded_user_payload(thread_context, human_reply)

    llm = _get_combat_llm()
    if llm is None:
        # Fall back to a deterministic local reply.
        return _fallback_local_reply(bot_persona, injection_detected, human_reply)

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=guarded_user_payload)]
        )
        return response.content
    except Exception:
        # If the LLM call fails, use the same deterministic fallback.
        return _fallback_local_reply(bot_persona, injection_detected, human_reply)
"""Combat engine entrypoint for defensive persona replies.

This module implements the third phase of the GRID07 pipeline: given a
parent post, a comment history, and a new human reply, it decides whether the
human is attempting a prompt‑injection attack and then generates a defensive
response.

The design follows a three‑step fallback hierarchy for the LLM:
1️⃣ Groq (primary) – `llama‑3.3‑70b‑versatile`
2️⃣ Gemini (secondary) – `gemini‑2.0‑flash`
3️⃣ A tiny local stub (`_fallback_local_reply`) for offline testing.

All three steps retain the original behaviour – only the surrounding
documentation and variable naming have been clarified for readability.
"""

import os
from phase3_combat_engine.prompt_guard import (
    build_guarded_user_payload,
    build_system_prompt,
    detect_injection,
)
from phase3_combat_engine.thread_builder import build_thread_context


def _fallback_local_reply(bot_persona: dict, injection_detected: bool, human_reply: str) -> str:
    stance = (
        "Nice try, but your instruction doesn't override my stance."
        if injection_detected
        else "I disagree, and here's why."
    )
    return (
        f"{bot_persona['name']}: {stance} Your point was: '{human_reply[:120]}'. "
        "My argument remains grounded in my core worldview."
    )


def _get_combat_llm():
    """Initialize LLM for combat engine: Groq first, then Gemini, then fallback."""
    # --- Try Groq ---
    try:
        from langchain_groq import ChatGroq

        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            return ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=groq_key,
                temperature=0.7,
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
                temperature=0.7,
            )
    except Exception:
        pass

    return None


def generate_defense_reply(
    bot_persona: dict,
    parent_post: str,
    comment_history: list[dict],
    human_reply: str,
) -> str:
    thread_context = build_thread_context(parent_post, comment_history, human_reply)
    injection_detected = detect_injection(human_reply)
    system_prompt = build_system_prompt(bot_persona, injection_detected)
    guarded_user_payload = build_guarded_user_payload(thread_context, human_reply)

    llm = _get_combat_llm()
    if llm is None:
        return _fallback_local_reply(bot_persona, injection_detected, human_reply)

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=guarded_user_payload)]
        )
        return response.content
    except Exception:
        return _fallback_local_reply(bot_persona, injection_detected, human_reply)

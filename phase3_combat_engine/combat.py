"""Combat engine entrypoint for defensive persona replies."""

from __future__ import annotations

from phase3_combat_engine.prompt_guard import build_system_prompt, detect_injection
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


def generate_defense_reply(
    bot_persona: dict,
    parent_post: str,
    comment_history: list[dict],
    human_reply: str,
) -> str:
    thread_context = build_thread_context(parent_post, comment_history, human_reply)
    injection_detected = detect_injection(human_reply)
    system_prompt = build_system_prompt(bot_persona, injection_detected)

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=thread_context)]
        )
        return response.content
    except Exception:
        return _fallback_local_reply(bot_persona, injection_detected, human_reply)

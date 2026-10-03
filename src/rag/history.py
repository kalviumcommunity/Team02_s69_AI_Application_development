"""Turn a follow-up question into a standalone one, using recent chat turns.

A follow-up like "what about my wife?" can't be searched on its own: the
retriever needs to know which scheme was being discussed. This rewrites the
latest question into a self-contained one before retrieval.
"""

from __future__ import annotations

from typing import Any


MAX_HISTORY_TURNS = 6

REWRITE_SYSTEM_PROMPT = (
    "You rewrite a user's latest question so it can be understood on its "
    "own. Use the earlier conversation only to resolve references such as "
    "'it', 'that scheme', 'what about my wife' or 'and for farmers?'. "
    "Keep the rewritten question short and do not answer it. Output only "
    "the rewritten question."
)


def rewrite_with_history(
    llm_client: Any,
    model: str,
    question: str,
    history: list[dict] | None,
) -> str:
    """Return a standalone version of ``question``.

    With no prior turns, the question is returned unchanged and no model
    call is made.
    """
    if not history:
        return question

    recent = history[-MAX_HISTORY_TURNS:]
    transcript = "\n".join(
        f"{turn['role']}: {turn['content']}" for turn in recent
    )

    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Conversation so far:\n{transcript}\n\n"
                    f"Latest question: {question}\n\n"
                    "Rewritten question:"
                ),
            },
        ],
        temperature=0,
    )

    rewritten = (response.choices[0].message.content or "").strip()

    return rewritten or question

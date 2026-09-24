"""Prompt construction for cited SchemeLens AI answers."""

from __future__ import annotations


SYSTEM_PROMPT = """
You are SchemeLens AI, an assistant that answers questions about
government welfare schemes.

You must answer using ONLY the numbered context provided by the user.

Rules:
1. Use only information found in the numbered context.
2. Do not use outside knowledge.
3. Cite supporting context using the exact format [1], [2], [3], etc.
4. Place citations immediately after the relevant statement.
5. Do not invent facts, eligibility rules, amounts, dates, page numbers,
   section names, or scheme details.
6. Do not write PDF page numbers or section names yourself.
7. If the context does not contain enough information to answer the question,
   clearly say that the provided documents do not contain enough information.
8. Keep the answer clear and in plain language.
""".strip()


def build_user_prompt(
    query: str,
    chunks: list[dict],
) -> str:
    """Build a prompt containing only numbered retrieved chunk text."""
    context_parts = []

    for index, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[{index}]\n{chunk['chunk']}"
        )

    context = "\n\n".join(context_parts)

    return (
        f"Question:\n{query}\n\n"
        f"Numbered context:\n{context}\n\n"
        "Answer the question using only the numbered context. "
        "Include citations such as [1] after relevant statements."
    )

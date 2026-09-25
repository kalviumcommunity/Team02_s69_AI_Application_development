"""Command-line interface for asking SchemeLens AI questions."""

from __future__ import annotations

import argparse

from src.rag.answerer import answer_question


def main() -> None:
    """Read a question from the command line and print the answer."""
    parser = argparse.ArgumentParser(
        description="Ask SchemeLens AI a question."
    )

    parser.add_argument(
        "question",
        nargs="+",
        help="Question to ask about the government schemes.",
    )

    args = parser.parse_args()

    query = " ".join(args.question)

    result = answer_question(query)

    print("\nAnswer:")
    print(result["answer"])

    print("\nCitations:")

    if not result["citations"]:
        print("No citations.")
        return

    for citation in result["citations"]:
        print(
            f"- {citation['scheme_name']} | "
            f"Page {citation['page']} | "
            f"{citation['section']}"
        )
        print(f"  {citation['snippet']}")


if __name__ == "__main__":
    main()

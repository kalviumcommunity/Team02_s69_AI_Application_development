"""Retrieval and refusal check against the curated question set.

For each in-corpus question, checks whether the expected scheme appears
among the top-k retrieved chunks (retrieval precision, not full answer
correctness). For out-of-corpus questions, checks that the system refuses
(``insufficient_information``) instead of guessing.

Run with: python -m evaluation.run_eval
Needs both EMBEDDING_API_KEY and CHAT_API_KEY configured (the latter only
for the out-of-corpus refusal check, which calls answer_question()).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.indexing.embedder import OpenAIEmbedder
from src.indexing.vector_store import VectorStore
from src.rag.answer import answer_question

QUESTIONS_FILE = Path(__file__).resolve().parent / "questions.csv"
RESULTS_FILE = Path(__file__).resolve().parent / "results.json"

TOP_K = 5


def load_questions(path: Path = QUESTIONS_FILE) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def evaluate_retrieval(question: dict, vector_store: VectorStore) -> dict:
    """Check whether the expected scheme appears in the top-k results."""
    results = vector_store.search(question["question_text"], top_k=TOP_K)
    retrieved_scheme_ids = [r["metadata"]["scheme_id"] for r in results]

    hit = question["expected_scheme_id"] in retrieved_scheme_ids

    return {
        "question_id": question["question_id"],
        "question_text": question["question_text"],
        "category": question["category"],
        "expected_scheme_id": question["expected_scheme_id"],
        "retrieved_scheme_ids": retrieved_scheme_ids,
        "hit": hit,
    }


def evaluate_refusal(question: dict) -> dict:
    """Check that an out-of-corpus question is refused, not guessed at."""
    response = answer_question(question["question_text"])
    refused = response["status"] == "insufficient_information"

    return {
        "question_id": question["question_id"],
        "question_text": question["question_text"],
        "category": question["category"],
        "refused": refused,
        "status": response["status"],
    }


def main() -> None:
    questions = load_questions()

    in_corpus = [q for q in questions if q["category"] != "out-of-corpus"]
    out_of_corpus = [q for q in questions if q["category"] == "out-of-corpus"]

    embedder = OpenAIEmbedder()
    vector_store = VectorStore(embedder=embedder)

    print(f"Loaded {len(questions)} questions "
          f"({len(in_corpus)} in-corpus, {len(out_of_corpus)} out-of-corpus).\n")

    print("Checking retrieval for in-corpus questions...")
    retrieval_results = [
        evaluate_retrieval(q, vector_store) for q in in_corpus
    ]

    print("Checking refusal for out-of-corpus questions...")
    refusal_results = [evaluate_refusal(q) for q in out_of_corpus]

    hits = sum(1 for r in retrieval_results if r["hit"])
    refusals = sum(1 for r in refusal_results if r["refused"])

    hit_rate = hits / len(retrieval_results) if retrieval_results else 0.0
    refusal_rate = (
        refusals / len(refusal_results) if refusal_results else 0.0
    )

    print("\nResults")
    print("-------")
    print(f"Retrieval hit rate:  {hits}/{len(retrieval_results)} "
          f"({hit_rate:.0%}) — target from the PRD: >=85%")
    print(f"Refusal rate:        {refusals}/{len(refusal_results)} "
          f"({refusal_rate:.0%}) — target: 100% (must never guess)")

    failed_retrieval = [r for r in retrieval_results if not r["hit"]]
    failed_refusal = [r for r in refusal_results if not r["refused"]]

    if failed_retrieval:
        print("\nFailed retrieval:")
        for r in failed_retrieval:
            print(
                f"  [{r['question_id']}] expected {r['expected_scheme_id']!r}, "
                f"got {r['retrieved_scheme_ids']}"
            )

    if failed_refusal:
        print("\nFailed refusal (answered instead of refusing):")
        for r in failed_refusal:
            print(f"  [{r['question_id']}] status={r['status']!r}")

    output = {
        "hit_rate": hit_rate,
        "refusal_rate": refusal_rate,
        "retrieval_results": retrieval_results,
        "refusal_results": refusal_results,
    }

    with RESULTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"\nSaved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()

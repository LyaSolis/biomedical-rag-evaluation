from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ollama import chat


MODEL_NAME = "qwen3:8b"


@dataclass
class RetrievedContext:
    text: str
    pubid: str
    chunk_index: int
    score: float | None = None


def build_context(contexts: Iterable[RetrievedContext]) -> str:
    """Format ranked retrieved passages with stable source identifiers."""
    sections = []

    for rank, context in enumerate(contexts, start=1):
        sections.append(
            f"[Source {rank} | PubMed ID: {context.pubid} | "
            f"Chunk: {context.chunk_index}]\n{context.text}"
        )

    return "\n\n".join(sections)


def build_prompt(question: str, context: str) -> str:
    """Build a grounded RAG prompt."""
    return f"""Answer the question using only the provided biomedical context.

Rules:
- Use the retrieved context as the evidence base.
- Do not invent facts that are not supported by the context.
- If the context does not contain enough information to answer the question, say so.
- Give a concise, evidence-grounded answer.
- Do not cite sources that are not present in the context.
- Refer to the relevant PubMed IDs when useful.

Question:
{question}

Retrieved biomedical context:
{context}
"""


def generate_answer(
    question: str,
    contexts: Iterable[RetrievedContext],
    model_name: str = MODEL_NAME,
) -> str:
    """Generate a grounded answer from ranked retrieved contexts."""
    context = build_context(contexts)
    prompt = build_prompt(question, context)

    response = chat(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a biomedical question-answering assistant. "
                    "Answer only from the supplied evidence. "
                    "Do not use outside knowledge."
                ),
            },
            {
                "role": "user",
                "content": prompt + "\n/no_think",
            },
        ],
        options={
            "temperature": 0.0,
        },
    )

    return response.message.content

from dataclasses import dataclass
from ollama import chat

MODEL_NAME = "qwen3:8b"

@dataclass
class RetrievedContext:
    text: str
    pubid: str
    chunk_index: int
    score: float | None = None

def build_context(contexts):
    return "\n\n".join(
        f"[Source {rank} | PubMed ID: {context.pubid} | "
        f"Chunk: {context.chunk_index}]\n{context.text}"
        for rank, context in enumerate(contexts, start=1)
    )

def build_prompt(question, context):
    return f"""Answer the question using only the provided biomedical context.

Rules:
- Use the retrieved context as the evidence base.
- Do not invent facts that are not supported by the context.
- If the context does not contain enough information to answer the question, say so.
- Give a concise, evidence-grounded answer.
- Refer to relevant PubMed IDs when useful.

Question:
{question}

Retrieved biomedical context:
{context}
"""

def generate_answer(question, contexts, model_name=MODEL_NAME):
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
                "content": build_prompt(question, build_context(contexts)) + "\n/no_think",
            },
        ],
        options={"temperature": 0.0},
    )
    return response.message.content

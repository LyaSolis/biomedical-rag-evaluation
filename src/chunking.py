import json


INPUT_PATH = "data/pubmedqa.jsonl"
OUTPUT_PATH = "data/pubmedqa_chunks.jsonl"

CHUNK_SIZE = 150
OVERLAP_CONTEXTS = 1
MIN_FINAL_CHUNK_RATIO = 0.25


def load_documents():
    with open(INPUT_PATH) as f:
        records = [json.loads(line) for line in f]

    documents = []

    for record in records:
        documents.append({
            "pubid": record["pubid"],
            "question": record["question"],
            "long_answer": record["long_answer"],
            "final_decision": record["final_decision"],
            "contexts": record["context"]["contexts"],
        })

    return documents


def chunk_contexts(contexts, chunk_size=CHUNK_SIZE):
    chunks = []
    current_contexts = []
    current_words = 0

    for context in contexts:
        context_words = len(context.split())

        if current_contexts and current_words + context_words > chunk_size:
            chunks.append(" ".join(current_contexts))

            # Carry the final context into the next chunk as semantic overlap.
            current_contexts = current_contexts[-OVERLAP_CONTEXTS:]
            current_words = sum(len(c.split()) for c in current_contexts)

        current_contexts.append(context)
        current_words += context_words

    if current_contexts:
        chunks.append(" ".join(current_contexts))

    # Merge a very small final chunk with the previous chunk.
    if len(chunks) > 1:
        final_size = len(chunks[-1].split())

        if final_size < chunk_size * MIN_FINAL_CHUNK_RATIO:
            chunks[-2] = chunks[-2] + " " + chunks[-1]
            chunks.pop()

    return chunks


def build_chunks(documents):
    chunks = []

    for document in documents:
        document_chunks = chunk_contexts(document["contexts"])

        for chunk_index, text in enumerate(document_chunks):
            chunks.append({
                "text": text,
                "pubid": document["pubid"],
                "question": document["question"],
                "long_answer": document["long_answer"],
                "final_decision": document["final_decision"],
                "chunk_index": chunk_index,
                "chunk_word_count": len(text.split()),
            })

    return chunks


def save_chunks(chunks):
    with open(OUTPUT_PATH, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")


documents = load_documents()
chunks = build_chunks(documents)
save_chunks(chunks)

print(f"Documents: {len(documents)}")
print(f"Chunks: {len(chunks)}")
print(f"Average chunks/document: {len(chunks) / len(documents):.2f}")
print(f"Saved chunks to: {OUTPUT_PATH}")
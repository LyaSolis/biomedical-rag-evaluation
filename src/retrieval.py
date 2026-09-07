import json
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


CHUNKS_PATH = Path("data/pubmedqa_chunks.jsonl")
QDRANT_PATH = "data/qdrant"
COLLECTION_NAME = "pubmedqa"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_records():
    with CHUNKS_PATH.open() as f:
        return [json.loads(line) for line in f]


def retrieve(question, client, model, top_k=5):
    query_vector = model.encode(
        question,
        convert_to_numpy=True,
    ).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points

    return results


def print_results(question, results):
    print(f"\nQuestion: {question}\n")

    for rank, result in enumerate(results, start=1):
        payload = result.payload

        print(f"--- Result {rank} | score={result.score:.4f} ---")
        print(f"PubID: {payload['pubid']}")
        print(f"Chunk: {payload['chunk_index']}")
        print(f"Words: {payload['chunk_word_count']}")
        print(payload["text"][:500])
        print()


records = load_records()

# Use a known PubMedQA question as a retrieval test.
test_record = records[0]

model = SentenceTransformer(EMBEDDING_MODEL_NAME)
client = QdrantClient(path=QDRANT_PATH)

results = retrieve(
    test_record["question"],
    client,
    model,
    top_k=5,
)

print_results(test_record["question"], results)

client.close()

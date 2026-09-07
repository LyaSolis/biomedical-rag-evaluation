import json
from pathlib import Path

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


CHUNKS_PATH = Path("data/pubmedqa_chunks.jsonl")
QDRANT_PATH = "data/qdrant"
COLLECTION_NAME = "pubmedqa"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32


def load_chunks():
    with CHUNKS_PATH.open() as f:
        return [json.loads(line) for line in f]


def create_collection(client, vector_size):
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )


def build_points(chunks, embeddings):
    points = []

    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(
            models.PointStruct(
                id=index,
                vector=embedding.tolist(),
                payload={
                    "text": chunk["text"],
                    "pubid": chunk["pubid"],
                    "question": chunk["question"],
                    "long_answer": chunk["long_answer"],
                    "final_decision": chunk["final_decision"],
                    "chunk_index": chunk["chunk_index"],
                    "chunk_word_count": chunk["chunk_word_count"],
                },
            )
        )

    return points


chunks = load_chunks()

print(f"Loaded {len(chunks)} chunks")

model = SentenceTransformer(EMBEDDING_MODEL_NAME)

embeddings = model.encode(
    [chunk["text"] for chunk in chunks],
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
)

print(f"Embedding dimension: {embeddings.shape[1]}")

client = QdrantClient(path=QDRANT_PATH)

create_collection(client, embeddings.shape[1])

points = build_points(chunks, embeddings)

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
)

collection_info = client.get_collection(COLLECTION_NAME)

print(f"Collection: {COLLECTION_NAME}")
print(f"Vectors stored: {collection_info.points_count}")
print(f"Qdrant path: {QDRANT_PATH}")

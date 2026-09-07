import json
from pathlib import Path

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/pubmedqa_chunks.jsonl")
QDRANT_PATH = "data/qdrant_pubmedbert"
COLLECTION_NAME = "pubmedqa_pubmedbert"
EMBEDDING_MODEL_NAME = "neuml/pubmedbert-base-embeddings"
BATCH_SIZE = 32

def load_chunks():
    with CHUNKS_PATH.open() as f:
        return [json.loads(line) for line in f]

def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    client = QdrantClient(path=QDRANT_PATH)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=embeddings.shape[1],
            distance=models.Distance.COSINE,
        ),
    )

    points = [
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
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    info = client.get_collection(COLLECTION_NAME)

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Vectors stored: {info.points_count}")
    print(f"Qdrant path: {QDRANT_PATH}")
    client.close()

if __name__ == "__main__":
    main()

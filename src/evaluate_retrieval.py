import json
import statistics
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


CHUNKS_PATH = Path("data/pubmedqa_chunks.jsonl")
QDRANT_PATH = "data/qdrant"
COLLECTION_NAME = "pubmedqa"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks():
    with CHUNKS_PATH.open() as f:
        return [json.loads(line) for line in f]


def evaluate_retrieval(chunks, client, model, k_values=(1, 3, 5)):
    questions = {}
    for chunk in chunks:
        questions[chunk["pubid"]] = chunk["question"]

    pubids = list(questions.keys())

    embeddings = model.encode(
        [questions[pubid] for pubid in pubids],
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    hits = {k: 0 for k in k_values}
    reciprocal_ranks = []

    for pubid, query_vector in zip(pubids, embeddings):
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector.tolist(),
            limit=max(k_values),
        ).points

        retrieved_pubids = [result.payload["pubid"] for result in results]

        rank = None

        for index, retrieved_pubid in enumerate(retrieved_pubids, start=1):
            if retrieved_pubid == pubid:
                rank = index
                break

        if rank is not None:
            reciprocal_ranks.append(1 / rank)
            for k in k_values:
                if rank <= k:
                    hits[k] += 1
        else:
            reciprocal_ranks.append(0)

    total = len(pubids)

    metrics = {
        f"Recall@{k}": hits[k] / total
        for k in k_values
    }

    metrics["MRR"] = statistics.mean(reciprocal_ranks)

    return metrics


chunks = load_chunks()

# One question per source document.
questions = {}
for chunk in chunks:
    questions[chunk["pubid"]] = chunk["question"]

print(f"Source documents evaluated: {len(questions)}")
print(f"Indexed chunks: {len(chunks)}")

model = SentenceTransformer(EMBEDDING_MODEL_NAME)
client = QdrantClient(path=QDRANT_PATH)

metrics = evaluate_retrieval(chunks, client, model)

print("\nRetrieval results:")
for name, value in metrics.items():
    print(f"{name}: {value:.3f}")

client.close()

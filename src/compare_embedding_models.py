import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/pubmedqa_chunks.jsonl")
MODEL_NAMES = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "neuml/pubmedbert-base-embeddings",
]
TOP_K_VALUES = [1, 3, 5]


def load_chunks():
    with CHUNKS_PATH.open() as f:
        return [json.loads(line) for line in f]


def mean_reciprocal_rank(ranks):
    reciprocal_ranks = [1.0 / rank for rank in ranks if rank is not None]
    return sum(reciprocal_ranks) / len(ranks)


def evaluate_model(model_name, documents, queries):
    print(f"\nLoading model: {model_name}")
    model = SentenceTransformer(model_name)

    document_texts = [record["text"] for record in documents]
    query_texts = [record["question"] for record in queries]
    query_pubids = [record["pubid"] for record in queries]

    document_embeddings = model.encode(
        document_texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    query_embeddings = model.encode(
        query_texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # With normalized embeddings, matrix multiplication is cosine similarity.
    scores = query_embeddings @ document_embeddings.T

    ranks = []
    recalls = {k: 0 for k in TOP_K_VALUES}

    for row_index, target_pubid in enumerate(query_pubids):
        ranked_indices = np.argsort(-scores[row_index])

        rank = None
        for position, document_index in enumerate(ranked_indices, start=1):
            if documents[document_index]["pubid"] == target_pubid:
                rank = position
                break

        ranks.append(rank)

        for k in TOP_K_VALUES:
            if rank is not None and rank <= k:
                recalls[k] += 1

    total = len(queries)
    metrics = {
        "model": model_name,
        "documents": len(documents),
        "queries": total,
        "embedding_dimension": int(document_embeddings.shape[1]),
        "recall_at_1": recalls[1] / total,
        "recall_at_3": recalls[3] / total,
        "recall_at_5": recalls[5] / total,
        "mrr": mean_reciprocal_rank(ranks),
        "top_5_misses": sum(rank is None or rank > 5 for rank in ranks),
    }

    return metrics


documents = load_chunks()

# One query per PubMed source, matching the existing benchmark.
queries_by_pubid = {}
for record in documents:
    queries_by_pubid.setdefault(record["pubid"], record)

queries = list(queries_by_pubid.values())

results = []
for model_name in MODEL_NAMES:
    results.append(evaluate_model(model_name, documents, queries))

print("\n=== Embedding Model Comparison ===")
print(
    f"{'Model':45} "
    f"{'Dim':>5} "
    f"{'R@1':>8} "
    f"{'R@3':>8} "
    f"{'R@5':>8} "
    f"{'MRR':>8} "
    f"{'Misses':>8}"
)
for result in results:
    print(
        f"{result['model'][:45]:45} "
        f"{result['embedding_dimension']:5d} "
        f"{result['recall_at_1']:.3f} "
        f"{result['recall_at_3']:.3f} "
        f"{result['recall_at_5']:.3f} "
        f"{result['mrr']:.3f} "
        f"{result['top_5_misses']:8d}"
    )

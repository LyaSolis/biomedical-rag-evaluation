import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

CHUNKS_PATH = Path("data/pubmedqa_chunks.jsonl")

EMBEDDING_MODEL = "neuml/pubmedbert-base-embeddings"
RERANKER_MODEL = "neuml/biomedbert-base-reranker"

CANDIDATE_K_VALUES = [5, 10, 20, 50]


def load_chunks():
    with CHUNKS_PATH.open() as f:
        return [json.loads(line) for line in f]


def mean_reciprocal_rank(ranks):
    return sum(
        1.0 / rank if rank is not None else 0.0
        for rank in ranks
    ) / len(ranks)


def source_rank(ranked_indices, documents, target_pubid):
    for rank, index in enumerate(ranked_indices, start=1):
        if documents[index]["pubid"] == target_pubid:
            return rank
    return None


documents = load_chunks()

# One query per source document, matching the existing benchmark.
queries_by_pubid = {}
for record in documents:
    queries_by_pubid.setdefault(record["pubid"], record)
queries = list(queries_by_pubid.values())

texts = [record["text"] for record in documents]
questions = [record["question"] for record in queries]
target_pubids = [record["pubid"] for record in queries]

print(f"Documents/chunks: {len(documents)}")
print(f"Queries: {len(queries)}")

print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
embedder = SentenceTransformer(EMBEDDING_MODEL)

document_embeddings = embedder.encode(
    texts,
    batch_size=32,
    normalize_embeddings=True,
    show_progress_bar=True,
)
query_embeddings = embedder.encode(
    questions,
    batch_size=32,
    normalize_embeddings=True,
    show_progress_bar=True,
)

similarities = query_embeddings @ document_embeddings.T

print(f"\nLoading reranker: {RERANKER_MODEL}")
reranker = CrossEncoder(RERANKER_MODEL)

results = []

for candidate_k in CANDIDATE_K_VALUES:
    ranks = []

    print(f"\nReranking top {candidate_k} candidates...")

    for query_index, question in enumerate(questions):
        candidate_indices = np.argsort(-similarities[query_index])[:candidate_k]

        pairs = [
            [question, documents[index]["text"]]
            for index in candidate_indices
        ]

        rerank_scores = reranker.predict(
            pairs,
            batch_size=16,
            show_progress_bar=False,
        )

        reranked_order = np.argsort(-np.asarray(rerank_scores))
        reranked_indices = candidate_indices[reranked_order]

        rank = source_rank(
            reranked_indices,
            documents,
            target_pubids[query_index],
        )
        ranks.append(rank)

    total = len(ranks)

    metrics = {
        "candidate_k": candidate_k,
        "recall_at_1": sum(r is not None and r <= 1 for r in ranks) / total,
        "recall_at_3": sum(r is not None and r <= 3 for r in ranks) / total,
        "recall_at_5": sum(r is not None and r <= 5 for r in ranks) / total,
        "mrr": mean_reciprocal_rank(ranks),
        "top_5_misses": sum(r is None or r > 5 for r in ranks),
    }

    results.append(metrics)

print("\n=== Reranker Results ===")
print(
    f"{'Candidates':>10} "
    f"{'R@1':>8} "
    f"{'R@3':>8} "
    f"{'R@5':>8} "
    f"{'MRR':>8} "
    f"{'Misses':>8}"
)

for result in results:
    print(
        f"{result['candidate_k']:>10} "
        f"{result['recall_at_1']:.3f} "
        f"{result['recall_at_3']:.3f} "
        f"{result['recall_at_5']:.3f} "
        f"{result['mrr']:.3f} "
        f"{result['top_5_misses']:>8}"
    )

output_path = Path("data/reranker_results.json")
with output_path.open("w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved: {output_path}")

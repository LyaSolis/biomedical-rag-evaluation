import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/pubmedqa_chunks.jsonl")
FAILURES_PATH = Path("data/retrieval_failures.json")

MODELS = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "PubMedBERT": "neuml/pubmedbert-base-embeddings",
}


def load_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def load_failures(path):
    with path.open() as f:
        return json.load(f)


def rank_source(scores, documents, target_pubid):
    ranked_indices = np.argsort(-scores)
    for rank, index in enumerate(ranked_indices, start=1):
        if documents[index]["pubid"] == target_pubid:
            return rank
    return None


documents = load_jsonl(CHUNKS_PATH)
failures = load_failures(FAILURES_PATH)

# The failure file contains one entry per query with its target source.
failure_queries = [
    {
        "pubid": failure["pubid"],
        "question": failure["question"],
    }
    for failure in failures
]

document_texts = [record["text"] for record in documents]

model_results = {}

for label, model_name in MODELS.items():
    print(f"\nLoading {label}: {model_name}")
    model = SentenceTransformer(model_name)

    document_embeddings = model.encode(
        document_texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    query_embeddings = model.encode(
        [item["question"] for item in failure_queries],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    scores = query_embeddings @ document_embeddings.T

    rows = []
    for i, failure in enumerate(failure_queries):
        rank = rank_source(
            scores[i],
            documents,
            failure["pubid"],
        )
        top5 = np.argsort(-scores[i])[:5]

        rows.append(
            {
                "pubid": failure["pubid"],
                "question": failure["question"],
                "correct_rank": rank,
                "top5_pubids": [
                    documents[index]["pubid"] for index in top5
                ],
            }
        )

    model_results[label] = rows


print("\n=== Failure Case Comparison ===")
print(
    f"{'PubID':>10}  {'MiniLM':>7}  {'PubMedBERT':>10}  Question"
)
print("-" * 100)

for i, failure in enumerate(failure_queries):
    mini_rank = model_results["MiniLM"][i]["correct_rank"]
    bio_rank = model_results["PubMedBERT"][i]["correct_rank"]

    print(
        f"{failure['pubid']:>10}  "
        f"{str(mini_rank):>7}  "
        f"{str(bio_rank):>10}  "
        f"{failure['question']}"
    )

mini_ranks = [row["correct_rank"] for row in model_results["MiniLM"]]
bio_ranks = [row["correct_rank"] for row in model_results["PubMedBERT"]]

def rank_category(rank):
    if rank is None or rank > 5:
        return "miss"
    return str(rank)

print("\n=== Changes ===")
for i, failure in enumerate(failure_queries):
    old_rank = mini_ranks[i]
    new_rank = bio_ranks[i]

    if old_rank != new_rank:
        print(
            f"{failure['pubid']}: "
            f"{rank_category(old_rank)} -> {rank_category(new_rank)}"
        )

# Save a machine-readable comparison for the notebook and future analysis.
output = []
for i, failure in enumerate(failure_queries):
    output.append(
        {
            "pubid": failure["pubid"],
            "question": failure["question"],
            "minilm_rank": mini_ranks[i],
            "pubmedbert_rank": bio_ranks[i],
            "change": (
                None
                if mini_ranks[i] == bio_ranks[i]
                else [mini_ranks[i], bio_ranks[i]]
            ),
        }
    )

output_path = Path("data/embedding_failure_case_comparison.json")
with output_path.open("w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved: {output_path}")

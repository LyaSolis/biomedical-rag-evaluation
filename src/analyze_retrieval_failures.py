import json
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


CHUNKS_PATH = Path("data/pubmedqa_chunks.jsonl")
QDRANT_PATH = "data/qdrant"
COLLECTION_NAME = "pubmedqa"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5


def load_chunks():
    with CHUNKS_PATH.open() as f:
        return [json.loads(line) for line in f]


def main():
    chunks = load_chunks()

    # One evaluation question per source document.
    questions = {}
    for chunk in chunks:
        questions[chunk["pubid"]] = chunk["question"]

    pubids = list(questions.keys())

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = QdrantClient(path=QDRANT_PATH)

    embeddings = model.encode(
        [questions[pubid] for pubid in pubids],
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    failures = []

    for pubid, query_vector in zip(pubids, embeddings):
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector.tolist(),
            limit=TOP_K,
        ).points

        retrieved = [
            {
                "rank": rank,
                "pubid": result.payload["pubid"],
                "score": round(result.score, 4),
                "chunk_index": result.payload["chunk_index"],
                "chunk_word_count": result.payload["chunk_word_count"],
                "text": result.payload["text"],
            }
            for rank, result in enumerate(results, start=1)
        ]

        correct_rank = next(
            (
                result["rank"]
                for result in retrieved
                if result["pubid"] == pubid
            ),
            None,
        )

        if correct_rank != 1:
            failures.append(
                {
                    "pubid": pubid,
                    "question": questions[pubid],
                    "correct_source_rank": correct_rank,
                    "retrieved": retrieved,
                }
            )

    output_path = Path("data/retrieval_failures.json")
    with output_path.open("w") as f:
        json.dump(failures, f, indent=2)

    print(f"Questions evaluated: {len(pubids)}")
    print(f"Rank-1 failures: {len(failures)}")
    print(f"Saved failures to: {output_path}")

    for failure in failures:
        print("\n" + "=" * 80)
        print(f"PubID: {failure['pubid']}")
        print(f"Correct source rank: {failure['correct_source_rank']}")
        print(f"Question: {failure['question']}")

        for result in failure["retrieved"]:
            print(
                f"  Rank {result['rank']}: "
                f"pubid={result['pubid']} "
                f"score={result['score']} "
                f"chunk={result['chunk_index']}"
            )

    client.close()


if __name__ == "__main__":
    main()

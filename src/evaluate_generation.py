import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from rag_generation import RetrievedContext, generate_answer
from reranking import BiomedicalReranker


QDRANT_PATH = "data/qdrant_pubmedbert"
COLLECTION_NAME = "pubmedqa_pubmedbert"
EMBEDDING_MODEL_NAME = "neuml/pubmedbert-base-embeddings"
DATA_PATH = "data/pubmedqa_chunks.jsonl"

CANDIDATE_K = 20
FINAL_CONTEXT_K = 5
DEFAULT_LIMIT = 100
RANDOM_SEED = 42


def load_evaluation_examples(data_path=DATA_PATH):
    """Recover the selected 500 questions from the chunk file."""
    examples = {}

    with open(data_path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            pubid = str(row["pubid"])

            if pubid not in examples:
                examples[pubid] = {
                    "pubid": pubid,
                    "question": row["question"],
                }

    dataset = load_dataset(
        "qiaojin/PubMedQA",
        "pqa_labeled",
        split="train",
    )

    references = {}
    for row in dataset:
        references[str(row["pubid"])] = {
            "long_answer": row["long_answer"],
            "final_decision": row["final_decision"],
        }

    evaluation_examples = []
    for pubid, example in examples.items():
        reference = references.get(pubid)
        if reference is None:
            raise ValueError(
                f"Could not find PubMedQA reference data for PubMed ID {pubid}."
            )

        evaluation_examples.append(
            {
                **example,
                **reference,
            }
        )

    evaluation_examples.sort(key=lambda row: row["pubid"])
    return evaluation_examples


def select_examples(examples, limit, seed):
    if limit <= 0 or limit >= len(examples):
        return examples

    rng = random.Random(seed)
    selected = rng.sample(examples, limit)
    selected.sort(key=lambda row: row["pubid"])
    return selected


def retrieve_and_rerank(question, embedder, reranker, client):
    query_vector = embedder.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    candidates = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=CANDIDATE_K,
    ).points

    reranked = reranker.rerank(
        question,
        candidates,
        top_k=FINAL_CONTEXT_K,
    )

    contexts = []
    for result in reranked:
        payload = result.payload
        contexts.append(
            RetrievedContext(
                text=payload["text"],
                pubid=str(payload["pubid"]),
                chunk_index=int(payload["chunk_index"]),
                score=float(result.score),
            )
        )

    return contexts


def build_record(example, contexts, answer):
    return {
        "pubid": example["pubid"],
        "question": example["question"],
        "reference_answer": example["long_answer"],
        "reference_decision": example["final_decision"],
        "generated_answer": answer,
        "retrieved_context": [
            {
                "rank": rank,
                "pubid": context.pubid,
                "chunk_index": context.chunk_index,
                "reranker_score": context.score,
                "text": context.text,
            }
            for rank, context in enumerate(contexts, start=1)
        ],
    }


def run_evaluation(limit=DEFAULT_LIMIT, seed=RANDOM_SEED, output_path=None):
    examples = load_evaluation_examples()
    selected = select_examples(examples, limit, seed)

    print(f"Evaluation pool: {len(examples)} examples")
    print(f"Examples selected: {len(selected)}")
    print(f"Candidate K: {CANDIDATE_K}")
    print(f"Final context K: {FINAL_CONTEXT_K}")

    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    reranker = BiomedicalReranker()
    client = QdrantClient(path=QDRANT_PATH)

    records = []

    try:
        for index, example in enumerate(selected, start=1):
            print(
                f"[{index}/{len(selected)}] "
                f"PubMed ID {example['pubid']}"
            )

            contexts = retrieve_and_rerank(
                example["question"],
                embedder,
                reranker,
                client,
            )

            answer = generate_answer(
                example["question"],
                contexts,
            )

            records.append(build_record(example, contexts, answer))

            if output_path:
                with open(output_path, "a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(records[-1], ensure_ascii=False) + "\n"
                    )

    finally:
        client.close()

    print(f"Completed: {len(records)} examples")

    if output_path:
        print(f"Saved results to: {output_path}")

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Run systematic RAG generation evaluation."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of examples to evaluate. Use 0 for all available examples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed used when selecting a subset.",
    )
    parser.add_argument(
        "--output",
        default="data/generation_evaluation.jsonl",
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Avoid accidentally appending to an earlier run.
    if output_path.exists():
        output_path.unlink()

    run_evaluation(
        limit=args.limit,
        seed=args.seed,
        output_path=str(output_path),
    )


if __name__ == "__main__":
    main()

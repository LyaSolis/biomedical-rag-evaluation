from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from rag_generation import RetrievedContext, generate_answer
from reranking import BiomedicalReranker

QDRANT_PATH = "data/qdrant_pubmedbert"
COLLECTION_NAME = "pubmedqa_pubmedbert"
EMBEDDING_MODEL_NAME = "neuml/pubmedbert-base-embeddings"

CANDIDATE_K = 20
FINAL_CONTEXT_K = 5

QUESTION = (
    "Do mitochondria play a role in the pathogenesis of "
    "non-alcoholic fatty liver disease?"
)

def main():
    print(f"Question: {QUESTION}")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    reranker = BiomedicalReranker()
    client = QdrantClient(path=QDRANT_PATH)

    query_vector = embedder.encode(
        QUESTION,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    candidates = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=CANDIDATE_K,
    ).points

    print(f"Retrieved {len(candidates)} candidates")

    reranked = reranker.rerank(
        QUESTION,
        candidates,
        top_k=FINAL_CONTEXT_K,
    )

    print("\n=== Ranked Retrieved Context ===")
    contexts = []

    for rank, result in enumerate(reranked, start=1):
        payload = result.payload
        print(
            f"\n--- Rank {rank} | PubMed ID: {payload['pubid']} "
            f"| Chunk: {payload['chunk_index']} "
            f"| Score: {result.score:.4f} ---"
        )
        print(payload["text"][:800])

        contexts.append(
            RetrievedContext(
                text=payload["text"],
                pubid=str(payload["pubid"]),
                chunk_index=int(payload["chunk_index"]),
                score=float(result.score),
            )
        )

    print("\n=== Generating Answer ===")
    answer = generate_answer(QUESTION, contexts)

    print("\n=== Answer ===")
    print(answer)

    client.close()

if __name__ == "__main__":
    main()

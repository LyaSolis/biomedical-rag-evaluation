from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME = "neuml/biomedbert-base-reranker"

class BiomedicalReranker:
    def __init__(self, model_name=RERANKER_MODEL_NAME):
        self.model = CrossEncoder(model_name)

    def rerank(self, question, results, top_k=5):
        if not results:
            return []

        pairs = [[question, result.payload["text"]] for result in results]
        scores = self.model.predict(
            pairs,
            batch_size=16,
            show_progress_bar=False,
        )

        ranked = sorted(
            zip(results, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        for result, score in ranked:
            result.score = float(score)

        return [result for result, _ in ranked[:top_k]]

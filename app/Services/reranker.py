import os
import httpx

COHERE_RERANK_URL = "https://api.cohere.com/v1/rerank"


class Reranker:
    def __init__(self, model_name=None, device=None):
        # device is unused (no local model to place on cuda/cpu) — kept for
        # call-site compatibility with the previous local cross-encoder API.
        self.model_name = model_name or os.getenv("RERANK_MODEL", "rerank-english-v3.0")
        self.api_key = os.getenv("COHERE_API_KEY")

    def rerank(self, query: str, passages: list[str], top_n=5, batch_size: int = 16):
        if not passages:
            return [], []

        if not self.api_key:
            raise RuntimeError("COHERE_API_KEY is not set in .env")

        # Pre-truncate to ~450 words so a single passage doesn't blow past
        # Cohere's per-document token limit.
        passages = [" ".join(p.split()[:450]) for p in passages]

        response = httpx.post(
            COHERE_RERANK_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model_name,
                "query": query,
                "documents": passages,
                "top_n": min(top_n, len(passages)),
            },
            timeout=30.0,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Cohere rerank request failed: {response.status_code} - {response.text}")

        results = response.json()["results"]
        top_indices = [r["index"] for r in results]
        scores = [r["relevance_score"] for r in results]
        reranked = [passages[i] for i in top_indices]
        return reranked, scores, top_indices

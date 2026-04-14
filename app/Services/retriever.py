from typing import List, Dict
import numpy as np
from sklearn.preprocessing import normalize

from app.Services import embedder, vector_db
from app.Services import bm25_index


def retrieve_top_chunks(query: str, top_k: int = 5) -> List[Dict]:
    """
    Embed the query, normalize it, and retrieve top-k most relevant document chunks.
    """
    query_embedding = embedder.embed_query(query)
    vector_db.ensure_loaded()
    results = vector_db.search(query_vector=query_embedding, top_k=top_k)

    # Empty results are a valid outcome (e.g. index has docs but none match
    # the query well enough).  Return [] so the caller can decide to fall back
    # to web search — no exception needed here.
    return results


def hybrid_retrieve(query: str, top_k: int = 5) -> List[Dict]:
    """
    Hybrid retrieval: combines FAISS dense search with BM25 keyword search
    via Reciprocal Rank Fusion (RRF).  Returns top_k results ordered by
    the fused score.

    RRF formula:  score(d) = Σ  1 / (k + rank_i(d))   for each retriever i
    with k=60 (standard constant that dampens high-rank dominance).
    """
    vector_db.ensure_loaded()
    if vector_db.index is None or vector_db.index.ntotal == 0:
        return []

    candidates = top_k * 5  # fetch more from each retriever for fusion

    # --- Dense (FAISS) ---
    query_embedding = embedder.embed_query(query)
    dense_results = vector_db.search(query_vector=query_embedding, top_k=candidates)

    # --- Sparse (BM25) ---
    bm25_index.ensure_built(vector_db.stored_chunks)
    bm25_results = bm25_index.search(query, top_k=candidates)

    # --- Reciprocal Rank Fusion ---
    RRF_K = 60
    fused_scores: Dict[int, float] = {}

    for rank, item in enumerate(dense_results):
        idx = item["index"]
        fused_scores[idx] = fused_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

    for rank, item in enumerate(bm25_results):
        idx = item["index"]
        fused_scores[idx] = fused_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

    # Sort by fused score descending
    sorted_indices = sorted(fused_scores.keys(), key=lambda i: -fused_scores[i])

    # Build result list matching vector_db.search() format
    results = []
    for idx in sorted_indices[:top_k]:
        if idx < len(vector_db.stored_chunks):
            results.append({
                "chunk": vector_db.stored_chunks[idx],
                "score": fused_scores[idx],
                "index": idx,
            })

    return results


def get_chunks_by_doc_ids(doc_ids: List[str]) -> Dict[str, List[str]]:
    """
    Return a dictionary mapping doc_id → list of chunk contents.
    Supports both '[doc_id] chunk' and 'doc_id|chunk' formats.
    """
    vector_db.ensure_loaded()
    result = {}

    for chunk_text in vector_db.stored_chunks:
        if "|" in chunk_text:
            doc_id, content = chunk_text.split("|", 1)
        elif chunk_text.startswith("[") and "]" in chunk_text:
            doc_id = chunk_text[1:chunk_text.index("]")]
            content = chunk_text[chunk_text.index("]") + 1:]
        else:
            continue

        doc_id = doc_id.strip()
        if doc_id in doc_ids:
            result.setdefault(doc_id, []).append(content.strip())

    return result

from typing import List, Dict
import numpy as np
from sklearn.preprocessing import normalize

from app.Services import embedder, vector_db


def retrieve_top_chunks(query: str, top_k: int = 5) -> List[Dict]:
    """
    Embed the query, normalize it, and retrieve top-k most relevant document chunks.
    """
    query_embedding = embedder.embed_query(query)
    vector_db.ensure_loaded()
    results = vector_db.search(query_vector=query_embedding, top_k=top_k)

    if not results:
        raise ValueError("Vector index not found or empty.")

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

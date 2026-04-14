"""
Lightweight BM25 keyword search index over stored chunks.

Rebuilds from the in-memory chunk list each time ensure_loaded() detects
a change.  For ~2K chunks this takes <50ms so persistence is unnecessary.
"""

import re
from typing import List, Dict
from rank_bm25 import BM25Okapi

_bm25: BM25Okapi | None = None
_raw_texts: List[str] = []
_last_n: int = 0


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer, lowercase."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _strip_prefix(chunk: str) -> str:
    if "|" in chunk:
        return chunk.split("|", 1)[1].strip()
    if chunk.startswith("[") and "]" in chunk:
        return chunk[chunk.index("]") + 1:].strip()
    return chunk.strip()


def ensure_built(stored_chunks: List[str]) -> None:
    """Rebuild BM25 index if the chunk list has grown."""
    global _bm25, _raw_texts, _last_n

    if len(stored_chunks) == _last_n and _bm25 is not None:
        return  # already up to date

    _raw_texts = [_strip_prefix(c) for c in stored_chunks]
    corpus = [_tokenize(t) for t in _raw_texts]
    _bm25 = BM25Okapi(corpus)
    _last_n = len(stored_chunks)


def search(query: str, top_k: int = 10) -> List[Dict]:
    """
    Return top_k results as list of dicts matching vector_db.search() format:
      {"chunk": <original tagged chunk>, "score": <bm25 score>, "index": <int>}
    Caller must call ensure_built() first.
    """
    if _bm25 is None or _last_n == 0:
        return []

    tokens = _tokenize(query)
    scores = _bm25.get_scores(tokens)

    # Get top_k indices by score (descending)
    top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    return [
        {"chunk": "", "score": float(scores[i]), "index": int(i)}
        for i in top_indices
        if scores[i] > 0
    ]

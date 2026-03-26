from typing import List
import re


def semantic_chunk_text(text: str, chunk_size: int = 8, overlap: int = 2) -> List[str]:
    """
    Splits text into overlapping chunks of `chunk_size` sentences.
    Default increased from 3 → 8 sentences so each chunk has sufficient
    context for the reranker to score it accurately.
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    # Filter out empty/whitespace-only sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= chunk_size:
        return [' '.join(sentences)]

    chunks = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i: i + chunk_size]
        chunks.append(' '.join(chunk))
        i += chunk_size - overlap

    return chunks

from typing import List
from sentence_transformers import SentenceTransformer, util

# Load model only once
model = SentenceTransformer("intfloat/e5-base-v2")  # You can switch to e5-large-v2 or any other

def semantic_chunk_text(text: str, chunk_size: int = 3, overlap: int = 1) -> List[str]:
    """
    Splits text into semantically meaningful chunks using sliding window.
    """
    # Step 1: Sentence split (basic)
    import re
    sentences = re.split(r'(?<=[.?!])\s+', text)
    if len(sentences) <= chunk_size:
        return [' '.join(sentences)]

    # Step 2: Chunking with overlap
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i:i + chunk_size]
        chunks.append(' '.join(chunk))
        i += chunk_size - overlap

    return chunks

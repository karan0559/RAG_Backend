from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
from sklearn.preprocessing import normalize
from app.Services import vector_db

model = SentenceTransformer("intfloat/e5-large-v2")

def embed_chunks(chunks: List[str], doc_id: str, metadata: Dict = None) -> np.ndarray:
    """
    Embeds and tags a list of text chunks with a doc_id.
    Format: "doc_id|chunk"
    """
    tagged_chunks = [f"{doc_id}|{chunk}" for chunk in chunks]
    prepared = [f"passage: {chunk}" for chunk in tagged_chunks]
    embeddings = model.encode(prepared, show_progress_bar=False, convert_to_numpy=True)
    embeddings = normalize(embeddings, axis=1)
    vector_db.add_embeddings(embeddings, tagged_chunks)
    return embeddings

def embed_query(query: str) -> np.ndarray:
    prepared = f"query: {query}"
    embedding = model.encode(prepared, show_progress_bar=False, convert_to_numpy=True)
    return normalize(embedding.reshape(1, -1), axis=1)[0]

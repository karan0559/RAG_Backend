import numpy as np
from typing import List, Dict
from sklearn.preprocessing import normalize
from app.Services import vector_db
from app.Services.model_loader import get_model

EMBED_MODEL_NAME = "intfloat/e5-large-v2"


def _get_model():
    return get_model(EMBED_MODEL_NAME)


def embed_chunks(chunks: List[str], doc_id: str, metadata: Dict = None) -> np.ndarray:
    """
    Embeds and tags a list of text chunks with a doc_id.
    Format stored: "doc_id|chunk"
    """
    model = _get_model()
    tagged_chunks = [f"{doc_id}|{chunk}" for chunk in chunks]
    # Use only the raw chunk text for embedding (not the tag prefix)
    prepared = [f"passage: {chunk}" for chunk in chunks]
    embeddings = model.encode(prepared, show_progress_bar=False, convert_to_numpy=True)
    embeddings = normalize(embeddings, axis=1)
    vector_db.add_embeddings(embeddings, tagged_chunks)
    return embeddings


def embed_query(query: str) -> np.ndarray:
    model = _get_model()
    prepared = f"query: {query}"
    embedding = model.encode(prepared, show_progress_bar=False, convert_to_numpy=True)
    return normalize(embedding.reshape(1, -1), axis=1)[0]

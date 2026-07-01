import os
import numpy as np
import httpx
from typing import List, Dict
from sklearn.preprocessing import normalize
from dotenv import load_dotenv
from app.Services import vector_db

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
COHERE_EMBED_URL = "https://api.cohere.com/v1/embed"
EMBED_MODEL_NAME = "embed-english-v3.0"

# Note: Cohere trial keys are rate-limited (see cohere.com for account
# management); replace with a production key if you hit quota errors.


def _cohere_embed(texts: List[str], input_type: str) -> np.ndarray:
    if not COHERE_API_KEY:
        raise RuntimeError("COHERE_API_KEY is not set in .env")

    response = httpx.post(
        COHERE_EMBED_URL,
        headers={"Authorization": f"Bearer {COHERE_API_KEY}"},
        json={
            "texts": texts,
            "model": EMBED_MODEL_NAME,
            "input_type": input_type,
            "embedding_types": ["float"],
        },
        timeout=30.0,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Cohere embed request failed: {response.status_code} - {response.text}")

    data = response.json()
    embeddings = data["embeddings"]["float"] if isinstance(data["embeddings"], dict) else data["embeddings"]
    return np.array(embeddings, dtype=np.float32)


def embed_chunks(chunks: List[str], doc_id: str, metadata: Dict = None) -> np.ndarray:
    """
    Embeds and tags a list of text chunks with a doc_id.
    Format stored: "doc_id|chunk"
    """
    tagged_chunks = [f"{doc_id}|{chunk}" for chunk in chunks]
    embeddings = _cohere_embed(chunks, input_type="search_document")
    embeddings = normalize(embeddings, axis=1)
    vector_db.add_embeddings(embeddings, tagged_chunks)
    return embeddings


def embed_query(query: str) -> np.ndarray:
    embedding = _cohere_embed([query], input_type="search_query")[0]
    return normalize(embedding.reshape(1, -1), axis=1)[0]

from sentence_transformers import SentenceTransformer
import chromadb
import os
import time
import uuid
from pathlib import Path
from app.Services.model_loader import get_model

# Absolute path — safe regardless of the directory uvicorn is launched from.
CHROMA_DIR = str(Path(__file__).resolve().parent.parent.parent / "data" / "chroma_memory")
os.makedirs(CHROMA_DIR, exist_ok=True)

EMBED_MODEL_NAME = "intfloat/e5-large-v2"

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection("chat_memory")


def _get_model():
    return get_model(EMBED_MODEL_NAME)


def embed(text: str):
    return _get_model().encode([f"query: {text}"])[0].tolist()


def add_to_memory(
    session_id: str,
    user_input: str,
    bot_output: str,
    source: str = "retriever",
    context_snippet: str = None
):
    metadata = {
        "session_id": session_id,
        "source": source,
        # Store creation time so we can sort by it later; ChromaDB does not
        # guarantee retrieval order, so we enforce it ourselves.
        "created_at": time.time(),
    }
    if context_snippet:
        metadata["context"] = context_snippet[:1000]

    collection.add(
        ids=[str(uuid.uuid4())],
        documents=[f"User: {user_input}\nAssistant: {bot_output}"],
        metadatas=[metadata],
        embeddings=[embed(user_input + bot_output)]
    )


def get_recent_history(session_id: str, limit: int = 10) -> str:
    """
    Retrieve the most recent `limit` conversation turns for a session in
    chronological order.

    ChromaDB's collection.get() does NOT guarantee insertion / time order, so
    we store a `created_at` unix timestamp in metadata at write time and sort
    here before slicing to `limit`.
    """
    try:
        results = collection.get(
            where={"session_id": session_id},
            limit=limit * 2,  # fetch extra to account for sorting + slicing
            include=["documents", "metadatas"],
        )
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        if not docs:
            return ""

        # Pair each document with its timestamp, sort ascending (oldest first),
        # then take the last `limit` turns so the most recent are always included.
        paired = sorted(
            zip(metas, docs),
            key=lambda x: float(x[0].get("created_at", 0)),
        )
        recent_docs = [doc for _, doc in paired[-limit:]]
        return "\n\n".join(recent_docs)
    except Exception:
        return ""


def clear_memory(session_id: str):
    all_ids = collection.get(where={"session_id": session_id}).get("ids", [])
    for chunk in [all_ids[i:i + 50] for i in range(0, len(all_ids), 50)]:
        collection.delete(ids=chunk)

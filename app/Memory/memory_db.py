from sentence_transformers import SentenceTransformer
import chromadb
import os
import uuid

CHROMA_DIR = "data/chroma_memory"
os.makedirs(CHROMA_DIR, exist_ok=True)

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection("chat_memory")

embedding_model = SentenceTransformer("intfloat/e5-large-v2")

def embed(text: str):
    return embedding_model.encode([f"query: {text}"])[0].tolist()

def add_to_memory(
    session_id: str,
    user_input: str,
    bot_output: str,
    source: str = "retriever",
    context_snippet: str = None
):
    metadata = {
        "session_id": session_id,
        "source": source
    }
    if context_snippet:
        metadata["context"] = context_snippet[:1000]

    collection.add(
        ids=[str(uuid.uuid4())],
        documents=[f"User: {user_input}\nAssistant: {bot_output}"],
        metadatas=[metadata],
        embeddings=[embed(user_input + bot_output)]
    )

def get_recent_history(session_id: str, limit: int = 30):
    results = collection.get(where={"session_id": session_id})
    docs = results.get("documents", [])[-limit:]
    return "\n\n".join(docs)

def clear_memory(session_id: str):
    all_ids = collection.get(where={"session_id": session_id}).get("ids", [])
    for chunk in [all_ids[i:i + 50] for i in range(0, len(all_ids), 50)]:
        collection.delete(ids=chunk)

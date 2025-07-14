from sentence_transformers import SentenceTransformer
import chromadb
import os
import uuid

# ✅ Setup ChromaDB storage path
CHROMA_DIR = "data/chroma_memory"
os.makedirs(CHROMA_DIR, exist_ok=True)

# ✅ Initialize Chroma Persistent Client (new syntax for Chroma v0.4+)
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection("chat_memory")

# ✅ Load embedding model
embedding_model = SentenceTransformer("intfloat/e5-large-v2")

# ✅ Embed text using E5 model
def embed(text: str):
    return embedding_model.encode([f"query: {text}"])[0].tolist()

# ✅ Add one interaction to memory (with optional source and context snippet)
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

# ✅ Fetch last N memory entries for a session
def get_recent_history(session_id: str, limit: int = 30):
    results = collection.get(where={"session_id": session_id})
    docs = results.get("documents", [])[-limit:]
    return "\n\n".join(docs)

# ✅ Optional: Clear memory for a session
def clear_memory(session_id: str):
    all_ids = collection.get(where={"session_id": session_id}).get("ids", [])
    for chunk in [all_ids[i:i + 50] for i in range(0, len(all_ids), 50)]:
        collection.delete(ids=chunk)

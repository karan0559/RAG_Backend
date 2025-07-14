import faiss
import numpy as np
import os
import pickle

DIM = 1024
INDEX_PATH = "data/index.faiss"
CHUNKS_PATH = "data/chunks.pkl"

index = None
stored_chunks = []

# Save FAISS index and chunks
def save_index(embeddings: np.ndarray, chunks: list[str]):
    global index, stored_chunks

    index = faiss.IndexFlatIP(DIM)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    stored_chunks = chunks
    print(f"✅ Saved index with {len(chunks)} vectors → {INDEX_PATH}")
    print(f"📝 Saved {len(chunks)} chunks → {CHUNKS_PATH}")

# Load FAISS index and chunks from disk
def load_index():
    global index, stored_chunks

    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            stored_chunks = pickle.load(f)
        print(f"📦 Loaded index ({index.ntotal} vectors) and {len(stored_chunks)} chunks.")
    else:
        index = faiss.IndexFlatIP(DIM)  
        stored_chunks = []
        print("⚠️ No existing index found. Start by uploading a document.")

# Ensure index is loaded before using
def ensure_loaded():
    global index
    if index is None or index.ntotal == 0:
        load_index()

# Add new embeddings and chunks
def add_embeddings(new_embeddings: np.ndarray, new_chunks: list[str]):
    global index, stored_chunks
    ensure_loaded()

    if index is None:
        index = faiss.IndexFlatIP(DIM)

    # Remove duplicates
    unique_chunks = []
    unique_embeddings = []
    for chunk, emb in zip(new_chunks, new_embeddings):
        if chunk not in stored_chunks:
            unique_chunks.append(chunk)
            unique_embeddings.append(emb)

    if unique_chunks:
        index.add(np.array(unique_embeddings))
        stored_chunks.extend(unique_chunks)

        faiss.write_index(index, INDEX_PATH)
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(stored_chunks, f)

        print(f"➕ Added {len(unique_chunks)} new unique chunks (total: {index.ntotal})")
    else:
        print("⚠️ No new unique chunks to add.")

# Search top-k similar chunks
def search(query_vector: np.ndarray, top_k=5):
    ensure_loaded()

    if index is None or index.ntotal == 0:
        print("❌ FAISS index is empty or not loaded.")
        return []

    D, I = index.search(np.array([query_vector]), top_k)
    results = []

    for idx, score in zip(I[0], D[0]):
        if idx < len(stored_chunks):
            results.append({
                "chunk": stored_chunks[idx],
                "score": float(score),
                "index": int(idx)
            })

    return results

# Expose metadata
def get_index_info():
    global index
    if index:
        return {"ntotal": index.ntotal, "dimension": index.d}
    return {"ntotal": 0, "dimension": 0}

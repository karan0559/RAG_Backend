import faiss
import numpy as np
import os
import pickle
from pathlib import Path

# Anchor paths to the project's data/ directory regardless of the working
# directory that uvicorn is launched from.
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

DIM = 1024
INDEX_PATH = str(_DATA_DIR / "index.faiss")
CHUNKS_PATH = str(_DATA_DIR / "chunks.pkl")

# Lightweight sidecar file: stores only the integer ntotal of the last-saved
# index.  ensure_loaded() reads this one tiny file to detect staleness instead
# of loading the full FAISS binary on every query (which was O(n_vectors*dim)).
INDEX_SIZE_PATH = str(_DATA_DIR / "index_size.txt")

index = None
stored_chunks = []


def _write_size(n: int) -> None:
    """Persist the current ntotal to the sidecar file so ensure_loaded() can
    detect whether the on-disk index has grown since the last in-memory load."""
    try:
        with open(INDEX_SIZE_PATH, "w") as f:
            f.write(str(n))
    except Exception:
        pass


def _read_disk_size() -> int:
    """Read ntotal from the sidecar file.  Returns 0 on any error so the
    caller falls back to a full load gracefully."""
    try:
        with open(INDEX_SIZE_PATH, "r") as f:
            return int(f.read().strip())
    except Exception:
        return 0


# Save FAISS index and chunks
def save_index(embeddings: np.ndarray, chunks: list[str]):
    global index, stored_chunks

    index = faiss.IndexFlatIP(DIM)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    _write_size(index.ntotal)  # keep sidecar in sync

    stored_chunks = chunks
    print(f"Saved index with {len(chunks)} vectors → {INDEX_PATH}")
    print(f" Saved {len(chunks)} chunks → {CHUNKS_PATH}")


# Load FAISS index and chunks from disk
def load_index():
    global index, stored_chunks

    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            stored_chunks = pickle.load(f)
        _write_size(index.ntotal)  # repair sidecar if it was absent
        print(f" Loaded index ({index.ntotal} vectors) and {len(stored_chunks)} chunks.")
    else:
        index = faiss.IndexFlatIP(DIM)
        stored_chunks = []
        _write_size(0)
        print(" No existing index found. Start by uploading a document.")


# Ensure index is loaded (and up-to-date) before using
def ensure_loaded():
    """
    Load the FAISS index from disk if:
      1. It has never been loaded (index is None or empty), OR
      2. The on-disk index has grown since the last in-memory load.

    Staleness is detected by reading index_size.txt — a tiny sidecar file
    written on every save/add.  This avoids loading the full FAISS binary on
    every query, which was expensive for large indexes.
    """
    global index

    in_mem_ntotal = index.ntotal if index is not None else 0

    if index is None or in_mem_ntotal == 0:
        load_index()
        return

    # Cheap staleness check: one small file read instead of a full binary load.
    disk_ntotal = _read_disk_size()
    if disk_ntotal > in_mem_ntotal:
        print(f" Index on disk ({disk_ntotal}) is ahead of memory ({in_mem_ntotal}) — reloading.")
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
        _write_size(index.ntotal)  # keep sidecar in sync after every add

        print(f" Added {len(unique_chunks)} new unique chunks (total: {index.ntotal})")
    else:
        print("No new unique chunks to add.")


# Search top-k similar chunks
def search(query_vector: np.ndarray, top_k=5):
    ensure_loaded()

    if index is None or index.ntotal == 0:
        print(" FAISS index is empty or not loaded.")
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

import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Set

# Protects all reads and writes to SESSION_DOCS_PATH so concurrent upload
# requests cannot interleave their file I/O and corrupt the JSON file.
_lock = threading.Lock()

# Absolute path — safe regardless of the directory uvicorn is launched from.
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SESSION_DOCS_PATH = str(_DATA_DIR / "session_docs.json")

# Sessions older than this many hours are automatically discarded so that stale
# document IDs do not pollute new, unrelated queries on the same machine.
SESSION_TTL_HOURS: int = int(os.getenv("SESSION_TTL_HOURS", "24"))
_TTL_SECONDS: float = SESSION_TTL_HOURS * 3600


def _ensure_parent_dir() -> None:
    os.makedirs(os.path.dirname(SESSION_DOCS_PATH), exist_ok=True)


def _load_map() -> Dict[str, dict]:
    """
    Read the session map from disk.  Must be called while holding _lock.

    Each entry has the shape:
        { "docs": ["doc_id", ...], "created_at": <unix timestamp> }
    """
    _ensure_parent_dir()
    if not os.path.exists(SESSION_DOCS_PATH):
        return {}

    try:
        with open(SESSION_DOCS_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return {}

        now = time.time()
        cleaned: Dict[str, dict] = {}
        for k, v in raw.items():
            # Support legacy format where the value was a plain list of doc IDs.
            if isinstance(v, list):
                v = {"docs": [str(d) for d in v if isinstance(d, str)], "created_at": now}
            if not isinstance(v, dict):
                continue
            created_at = float(v.get("created_at", now))
            # Drop sessions that have exceeded the TTL.
            if (now - created_at) > _TTL_SECONDS:
                continue
            docs = [str(d) for d in v.get("docs", []) if isinstance(d, str)]
            cleaned[str(k)] = {"docs": docs, "created_at": created_at}

        return cleaned
    except Exception:
        return {}


def _save_map(data: Dict[str, dict]) -> None:
    """Write the session map to disk. Must be called while holding _lock."""
    _ensure_parent_dir()
    with open(SESSION_DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def register_doc(session_id: str, doc_id: str) -> None:
    """Register a document ID under a session, creating the session entry if needed."""
    if not session_id or not doc_id:
        return

    with _lock:
        data = _load_map()
        entry = data.get(session_id, {"docs": [], "created_at": time.time()})
        existing: Set[str] = set(entry["docs"])
        existing.add(doc_id)
        entry["docs"] = sorted(existing)
        # Preserve the original creation timestamp so TTL is based on session start.
        data[session_id] = entry
        _save_map(data)


def get_docs(session_id: str) -> List[str]:
    """Return the list of doc IDs registered for a session (empty if expired or unknown)."""
    if not session_id:
        return []
    with _lock:
        entry = _load_map().get(session_id, {})
        return entry.get("docs", [])


def clear_session(session_id: str) -> None:
    """
    Explicitly remove all document associations for a session.
    Useful when the user clicks "New Chat" and the frontend resets its scope.
    """
    if not session_id:
        return
    with _lock:
        data = _load_map()
        data.pop(session_id, None)
        _save_map(data)

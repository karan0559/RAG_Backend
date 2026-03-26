import json
import os
from typing import Dict, List, Set


SESSION_DOCS_PATH = "data/session_docs.json"


def _ensure_parent_dir() -> None:
    os.makedirs(os.path.dirname(SESSION_DOCS_PATH), exist_ok=True)


def _load_map() -> Dict[str, List[str]]:
    _ensure_parent_dir()
    if not os.path.exists(SESSION_DOCS_PATH):
        return {}

    try:
        with open(SESSION_DOCS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {
                str(k): [str(v) for v in values if isinstance(v, str)]
                for k, values in data.items()
                if isinstance(values, list)
            }
    except Exception:
        pass

    return {}


def _save_map(data: Dict[str, List[str]]) -> None:
    _ensure_parent_dir()
    with open(SESSION_DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def register_doc(session_id: str, doc_id: str) -> None:
    if not session_id or not doc_id:
        return

    data = _load_map()
    existing: Set[str] = set(data.get(session_id, []))
    existing.add(doc_id)
    data[session_id] = sorted(existing)
    _save_map(data)


def get_docs(session_id: str) -> List[str]:
    if not session_id:
        return []
    return _load_map().get(session_id, [])

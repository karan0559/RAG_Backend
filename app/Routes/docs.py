from fastapi import APIRouter
from app.Services import vector_db
from app.Memory import session_docs

router = APIRouter()

@router.get("/list", summary="Get list of available document IDs")
async def list_doc_ids(session_id: str = None):
    vector_db.ensure_loaded()

    session_filter = set(session_docs.get_docs(session_id)) if session_id else None

    doc_ids = set()
    for chunk in vector_db.stored_chunks:
        if "|" in chunk:
            doc_id, _ = chunk.split("|", 1)
            doc_id = doc_id.strip()
            if session_filter is not None and doc_id not in session_filter:
                continue
            doc_ids.add(doc_id)

    return {"doc_ids": sorted(doc_ids)}

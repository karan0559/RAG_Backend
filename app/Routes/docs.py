from fastapi import APIRouter
from app.Services import vector_db

router = APIRouter()

@router.get("/list", summary="Get list of available document IDs")
async def list_doc_ids():
    vector_db.ensure_loaded()

    doc_ids = set()
    for chunk in vector_db.stored_chunks:
        if "|" in chunk:
            doc_id, _ = chunk.split("|", 1)
            doc_ids.add(doc_id.strip())

    return {"doc_ids": sorted(doc_ids)}

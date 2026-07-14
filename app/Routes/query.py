from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import traceback

from app.Services.query_pipeline import run_query

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    session_id: str = None
    top_k: int = 5
    tts: bool = False
    doc_ids: Optional[List[str]] = None


@router.post("/", summary="Ask a question over uploaded content")
async def query_rag(body: QueryRequest):
    try:
        return await run_query(
            query=body.query,
            session_id=body.session_id,
            top_k=body.top_k,
            tts=body.tts,
            doc_ids=body.doc_ids,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.Services import retriever, llm
import traceback

router = APIRouter()


class CompareRequest(BaseModel):
    doc_ids: list[str]
    mode: str = "compare"  # or "summarize"
    question: str = ""


@router.post("/compare/", summary="Compare or summarize multiple documents")
async def compare_or_summarize(request: CompareRequest):
    try:
        doc_chunks = retriever.get_chunks_by_doc_ids(request.doc_ids)

        if not doc_chunks:
            raise HTTPException(status_code=404, detail="No matching documents found.")

        if request.mode == "summarize":
            all_text = "\n\n".join(["\n".join(chunks) for chunks in doc_chunks.values()])
            result = llm.summarize_text(all_text)
            return {"mode": "summarize", "summary": result}

        else:
            result = llm.compare_documents(doc_chunks, question=request.question)
            return {"mode": "compare", "comparison": result}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

# app/Routes/query.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.Services import retriever, llm
from app.Services.reranker import Reranker
from app.Memory import memory_db
import traceback
import uuid

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    session_id: str = None  # ✅ Optional now
    top_k: int = 5

reranker = Reranker()

@router.post("/query/", summary="Ask a question over uploaded content")
async def query_rag(request: QueryRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())  # ✅ Auto-generate if not given
        print(f"🔍 Query: {request.query} | Session ID: {session_id}")
        
        # Step 1: Retrieve more than needed
        top_chunks = retriever.retrieve_top_chunks(request.query, top_k=10)
        if not top_chunks:
            return {
                "question": request.query,
                "session_id": session_id,
                "answer": "❌ No relevant documents found.",
                "top_chunks": []
            }

        raw_texts = [chunk["chunk"] for chunk in top_chunks]

        # Step 2: Rerank
        reranked_texts, scores = reranker.rerank(request.query, raw_texts, top_n=request.top_k)

        # Step 3: Match reranked to original
        final_chunks = []
        for reranked in reranked_texts:
            for original in top_chunks:
                if reranked == original["chunk"]:
                    final_chunks.append(original)
                    break

        # Step 4: Build context = memory + docs
        doc_context = "\n\n".join([chunk["chunk"] for chunk in final_chunks])
        chat_history = memory_db.get_recent_history(session_id, limit=20)
        combined_context = f"{chat_history}\n\n{doc_context}".strip()

        print(f"📚 Combined context length: {len(combined_context)} chars")

        # Step 5: Answer
        answer = llm.answer_question(context=combined_context, question=request.query)
        print(f"🧠 LLM Answer: {answer}")

        # Step 6: Add to memory
        memory_db.add_to_memory(session_id, user_input=request.query, bot_output=answer)

        return {
            "question": request.query,
            "session_id": session_id,
            "answer": answer,
            "top_chunks": final_chunks
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

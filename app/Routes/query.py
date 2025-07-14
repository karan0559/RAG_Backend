from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from app.Services import retriever, llm
from app.Services.reranker import Reranker
from app.Services import web_search
from app.Services.tts import generate_tts
from app.Memory import memory_db
from pathlib import Path
import traceback
import uuid

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    session_id: str = None
    top_k: int = 5
    tts: bool = False

reranker = Reranker()
RELEVANCE_THRESHOLD = 0.15

@router.post("/query/", summary="Ask a question over uploaded content")
async def query_rag(request: Request, body: QueryRequest):
    try:
        session_id = body.session_id or str(uuid.uuid4())
        print(f"\n🔍 Query: {body.query} | Session ID: {session_id}")

        fallback_used = False
        doc_context = ""
        final_chunks = []

        #Retrieve top chunks
        top_chunks = retriever.retrieve_top_chunks(body.query, top_k=10)
        print(f"📥 Retrieved {len(top_chunks)} chunks")

        if top_chunks:
            raw_texts = [chunk["chunk"] for chunk in top_chunks]
            reranked_texts, scores = reranker.rerank(body.query, raw_texts, top_n=body.top_k)

            final_scores = []
            for reranked in reranked_texts:
                for idx, original in enumerate(top_chunks):
                    if reranked == original["chunk"]:
                        final_chunks.append(original)
                        score = scores[reranked_texts.index(reranked)]
                        final_scores.append(score)
                        print(f"🔹 Score: {score:.3f} | Chunk: {reranked[:100]}...")
                        break

            if all(score < RELEVANCE_THRESHOLD for score in final_scores):
                fallback_used = True
                print("Fallback to Web Search (All chunks below threshold)")
                doc_context = web_search.search_web(body.query)
            else:
                doc_context = "\n\n".join([chunk["chunk"] for chunk in final_chunks])
        else:
            fallback_used = True
            print(" Fallback (No chunks retrieved at all)")
            doc_context = web_search.search_web(body.query)

        #Build context with memory
        chat_history = memory_db.get_recent_history(session_id, limit=20)
        combined_context = f"{chat_history}\n\n{doc_context}".strip()
        print(f"📚 Combined context length: {len(combined_context)} chars")

        #Get LLM answer
        answer = llm.answer_question(context=combined_context, question=body.query)
        print(f"🧠 LLM Answer: {answer}")

        #Save to memory
        source = "web_search" if fallback_used else "retriever"
        context_snippet = doc_context[:1000]
        memory_db.add_to_memory(
            session_id=session_id,
            user_input=body.query,
            bot_output=answer,
            source=source,
            context_snippet=context_snippet
        )

        #Generate TTS
        tts_path = None
        if body.tts:
            print(" Generating TTS...")
            try:
                tts_file = await generate_tts(answer)
                tts_filename = Path(tts_file).name.strip().replace('"', '').replace("'", "")
                tts_path = f"/audio/{tts_filename}"  
                print(f"🔈 TTS URL: {tts_path}")
            except Exception as tts_err:
                print(f" TTS generation failed: {tts_err}")
                tts_path = None

        return {
            "question": body.query,
            "session_id": session_id,
            "answer": answer,
            "top_chunks": final_chunks if not fallback_used else [],
            "fallback_used": fallback_used,
            "tts_audio_path": tts_path
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

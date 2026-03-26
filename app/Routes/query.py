from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from app.Services import retriever, llm
from app.Services.reranker import Reranker
from app.Services import web_search, vector_db, embedder
from app.Services.tts import generate_tts
from app.Memory import memory_db, session_docs
from pathlib import Path
import traceback
import uuid
import math
from typing import List, Optional
import numpy as np

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    session_id: str = None
    top_k: int = 5
    tts: bool = False
    doc_ids: Optional[List[str]] = None


reranker = Reranker()


def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def strip_prefix(chunk_text: str) -> str:
    """Remove 'doc_id|' storage tag from stored chunk text."""
    if "|" in chunk_text:
        return chunk_text.split("|", 1)[1]
    return chunk_text


def extract_doc_id(chunk_text: str) -> Optional[str]:
    if "|" not in chunk_text:
        return None
    return chunk_text.split("|", 1)[0].strip()


def retrieve_scoped_chunks(query: str, allowed_doc_ids: set[str], top_k: int = 20) -> List[dict]:
    """
    Retrieve top chunks only from allowed doc_ids by scoring against vectors
    reconstructed from the FAISS index.
    """
    vector_db.ensure_loaded()
    if vector_db.index is None or vector_db.index.ntotal == 0:
        return []

    query_vec = embedder.embed_query(query)
    scored: List[dict] = []

    for idx, chunk_text in enumerate(vector_db.stored_chunks):
        chunk_doc_id = extract_doc_id(chunk_text)
        if not chunk_doc_id or chunk_doc_id not in allowed_doc_ids:
            continue

        try:
            vec = vector_db.index.reconstruct(idx)
            score = float(np.dot(vec, query_vec))
            scored.append({"chunk": chunk_text, "score": score, "index": int(idx)})
        except Exception:
            continue

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


@router.post("/", summary="Ask a question over uploaded content")
async def query_rag(request: Request, body: QueryRequest):
    try:
        session_id = body.session_id or str(uuid.uuid4())
        print(f"\n📨 Query: {body.query!r} | Session: {session_id}")

        fallback_used = False
        doc_context = ""
        final_chunks = []
        allowed_doc_ids = set(body.doc_ids or [])
        if not allowed_doc_ids and session_id:
            allowed_doc_ids = set(session_docs.get_docs(session_id))
        has_doc_scope = len(allowed_doc_ids) > 0

        # ── Step 1: Check if the vector store has ANY documents ───────────
        vector_db.ensure_loaded()
        index_has_docs = (vector_db.index is not None and vector_db.index.ntotal > 0)
        print(f"  Index size: {vector_db.index.ntotal if vector_db.index else 0} vectors")

        if not index_has_docs:
            # No documents uploaded at all → use web search
            fallback_used = True
            print("  ⚠️  Index is empty → web search")
            doc_context = await web_search.search_web(body.query)
        else:
            # ── Step 2: Retrieve top-k chunks from FAISS ──────────────────
            try:
                retrieval_k = 10
                if allowed_doc_ids:
                    # Pull a wider candidate pool when scoping to doc IDs,
                    # then filter to the session/doc subset.
                    retrieval_k = max(50, body.top_k * 20)
                top_chunks = retriever.retrieve_top_chunks(body.query, top_k=retrieval_k)
            except Exception as e:
                print(f"  Retriever error: {e}")
                top_chunks = []

            print(f"  Retrieved {len(top_chunks)} chunks from vector store")

            if allowed_doc_ids:
                scoped_chunks = []
                for c in top_chunks:
                    chunk_doc_id = extract_doc_id(c.get("chunk", ""))
                    if chunk_doc_id and chunk_doc_id in allowed_doc_ids:
                        scoped_chunks.append(c)
                print(
                    f"  Session/doc scope active: {len(allowed_doc_ids)} docs | "
                    f"{len(scoped_chunks)} of {len(top_chunks)} chunks kept"
                )
                top_chunks = scoped_chunks

                if not top_chunks:
                    # If ANN top-k missed the session docs, do a strict in-scope pass.
                    top_chunks = retrieve_scoped_chunks(
                        body.query,
                        allowed_doc_ids=allowed_doc_ids,
                        top_k=max(20, body.top_k * 10),
                    )
                    print(f"  Scoped fallback retrieved {len(top_chunks)} chunks")

            if top_chunks:
                # ── Step 3: Strip doc_id prefix before reranking ──────────
                raw_texts_clean = [strip_prefix(c["chunk"]) for c in top_chunks]

                # ── Step 4: Rerank ────────────────────────────────────────
                try:
                    reranked_texts, raw_scores = reranker.rerank(
                        body.query, raw_texts_clean, top_n=body.top_k
                    )
                    norm_scores = [sigmoid(float(s)) for s in raw_scores]
                    print(f"  Reranker scores (sigmoid): {[round(s, 3) for s in norm_scores]}")
                except Exception as e:
                    print(f"  Reranker error: {e} — using FAISS order")
                    reranked_texts = raw_texts_clean[:body.top_k]
                    norm_scores = [1.0] * len(reranked_texts)

                # ── Step 5: Map back to original chunk objects ────────────
                clean_to_original = {}
                for i in range(len(top_chunks)):
                    clean_to_original.setdefault(raw_texts_clean[i], []).append(top_chunks[i])

                for reranked_text, norm_score in zip(reranked_texts, norm_scores):
                    candidates = clean_to_original.get(reranked_text, [])
                    original = candidates.pop(0) if candidates else None
                    if original:
                        final_chunks.append(original)
                        print(f"  🔹 Score={norm_score:.3f} | {reranked_text[:100]}")

                # ── Step 6: Build document context ────────────────────────
                # If document chunks were found, stay strictly document-first.
                doc_context = "\n\n".join(
                    strip_prefix(c["chunk"]) for c in final_chunks
                )
                fallback_used = False
            else:
                if has_doc_scope:
                    # Session has scoped documents; do not jump to web.
                    fallback_used = False
                    print("  ⚠️  No scoped document chunks found; skipping web fallback")
                    doc_context = "No relevant section found in the currently uploaded session documents."
                else:
                    # FAISS returned 0 results (shouldn't happen if index_has_docs)
                    fallback_used = True
                    print("  ⚠️  FAISS returned no results despite non-empty index → web search")
                    doc_context = await web_search.search_web(body.query)

        # ── Step 7: Build context and answer ─────────────────────────────
        chat_history = memory_db.get_recent_history(session_id, limit=10)
        combined_context = f"{chat_history}\n\n{doc_context}".strip()
        print(f"  📚 Context length: {len(combined_context)} chars | fallback={fallback_used}")

        answer = await llm.answer_question(
            context=combined_context,
            question=body.query,
            session_id=session_id,
        )
        print(f"  ✅ {answer[:120]}")

        # ── Step 8: Optional TTS ──────────────────────────────────────────
        tts_path = None
        if body.tts:
            try:
                tts_file = await generate_tts(answer)
                tts_path = f"/audio/{Path(tts_file).name}"
            except Exception as e:
                print(f"  TTS failed: {e}")

        return {
            "question": body.query,
            "session_id": session_id,
            "answer": answer,
            "top_chunks": final_chunks if not fallback_used else [],
            "fallback_used": fallback_used,
            "tts_audio_path": tts_path,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

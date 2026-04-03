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
import re
import time
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


def is_non_usable_chunk(chunk_text: str) -> bool:
    clean = strip_prefix(chunk_text).strip().lower()
    if not clean:
        return True
    bad_markers = (
        "ocr failed:",
        "extraction failed:",
        "unsupported file type",
        "unsupported parser output format",
        "no readable text found",
    )
    return any(marker in clean for marker in bad_markers)


def is_small_talk(query: str) -> bool:
    """Fast heuristic intent gate for greetings and short conversational turns."""
    normalized = re.sub(r"[^a-z0-9\s]", "", query.lower()).strip()
    if not normalized:
        return True

    small_talk_exact = {
        "hi",
        "hello",
        "hey",
        "yo",
        "hola",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "hows it going",
        "thanks",
        "thank you",
        "ok",
        "okay",
    }
    if normalized in small_talk_exact:
        return True

    small_talk_prefixes = (
        "hi ",
        "hello ",
        "hey ",
        "thanks ",
        "thank you ",
        "how are you ",
    )
    return normalized.startswith(small_talk_prefixes)


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
        t0 = time.perf_counter()
        session_id = body.session_id or str(uuid.uuid4())
        print(f"\n📨 Query: {body.query!r} | Session: {session_id}")

        fallback_used = False
        fallback_reason = None
        doc_context = ""
        final_chunks = []
        conversational = is_small_talk(body.query)
        allowed_doc_ids = set(body.doc_ids or [])
        if not allowed_doc_ids and session_id:
            allowed_doc_ids = set(session_docs.get_docs(session_id))
        has_doc_scope = len(allowed_doc_ids) > 0
        no_session_docs = bool(session_id) and not has_doc_scope

        if conversational:
            print("  💬 Small-talk intent detected → skipping retrieval/web fallback")
        elif no_session_docs:
            fallback_used = True
            fallback_reason = "no_session_docs"
            print("  ⚠️  No documents registered for this session → web search")
            doc_context = await web_search.search_web(body.query)
        else:
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
                # ── Step 2: Retrieve top-k chunks ──────────────────────────────────
                try:
                    if allowed_doc_ids:
                        # Directly retrieve chunks strictly scoped to the allowed docs.
                        # This avoids the sub-selection bug caused by post-filtering the global ANN search.
                        top_chunks = retrieve_scoped_chunks(
                            body.query,
                            allowed_doc_ids=allowed_doc_ids,
                            top_k=max(20, body.top_k * 10),
                        )
                        print(f"  Scoped search retrieved {len(top_chunks)} chunks for {len(allowed_doc_ids)} docs")
                    else:
                        retrieval_k = max(10, body.top_k * 4)
                        top_chunks = retriever.retrieve_top_chunks(body.query, top_k=retrieval_k)
                        print(f"  Global ANN retrieved {len(top_chunks)} chunks")
                except Exception as e:
                    print(f"  Retriever error: {e}")
                    top_chunks = []

                top_chunks = [c for c in top_chunks if not is_non_usable_chunk(c.get("chunk", ""))]
                print(f"  Kept {len(top_chunks)} chunks after quality filter")

                if top_chunks:
                    # ── Step 3: Strip doc_id prefix before reranking ──────────
                    raw_texts_clean = [strip_prefix(c["chunk"]) for c in top_chunks]

                    # ── Step 4: Rerank ────────────────────────────────────────
                    try:
                        # Cap reranking work to keep CPU latency predictable.
                        max_rerank_candidates = min(len(raw_texts_clean), max(12, body.top_k * 4))
                        rerank_candidates = raw_texts_clean[:max_rerank_candidates]
                        reranked_texts, raw_scores = reranker.rerank(
                            body.query, rerank_candidates, top_n=body.top_k
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
                        # If session docs exist but none are relevant/usable, answer from web.
                        fallback_used = True
                        fallback_reason = "no_relevant_session_docs"
                        print("  ⚠️  No scoped document chunks found; falling back to web search")
                        doc_context = await web_search.search_web(body.query)
                    else:
                        # FAISS returned 0 results (shouldn't happen if index_has_docs)
                        fallback_used = True
                        fallback_reason = "retrieval_empty"
                        print("  ⚠️  FAISS returned no results despite non-empty index → web search")
                        doc_context = await web_search.search_web(body.query)

        # ── Step 7: Build context and answer ─────────────────────────────
        combined_context = doc_context.strip()
        print(
            f"  📚 Context length: {len(combined_context)} chars | "
            f"fallback={fallback_used} | conversational={conversational}"
        )

        answer = await llm.answer_question(
            context=combined_context,
            question=body.query,
            session_id=session_id,
            use_documents=not conversational,
        )

        if fallback_used and not conversational:
            if fallback_reason == "no_session_docs":
                prefix = "No uploaded document was provided for this session. Answering from the web."
            elif fallback_reason == "no_relevant_session_docs":
                prefix = "No relevant section was found in uploaded documents. Answering from the web."
            else:
                prefix = "Document retrieval returned no usable context. Answering from the web."
            answer = f"{prefix}\n\n{answer}"

        print(f"  ⏱️ Query total: {time.perf_counter() - t0:.2f}s")
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

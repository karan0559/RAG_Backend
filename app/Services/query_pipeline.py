"""
Framework-agnostic core of the query/answer flow: small-talk detection,
hybrid/scoped retrieval, reranking, relevance-gated web fallback, LLM call,
and optional TTS.

Extracted out of app/Routes/query.py so both the FastAPI route and the
Streamlit app can call the exact same logic instead of maintaining two
copies of the relevance-gating behavior.
"""
import re
import time
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np

from app.Services import retriever, llm, web_search, vector_db, embedder
from app.Services.reranker import Reranker
from app.Services.tts import generate_tts
from app.Memory import session_docs

# Lazy singleton — loaded on first use so a missing/undownloaded model does
# not crash the app at startup.  The reranker error path below already
# falls back gracefully to FAISS order if reranking fails.
_reranker: Reranker | None = None


def _get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


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
    # Kept in sync with app/Services/extractor.py's _looks_like_error_text —
    # this is the query-time safety net for chunks that were indexed before
    # that filter existed or was tightened.
    bad_markers = (
        "ocr failed:",
        "extraction failed:",
        "pdf parsing failed:",
        "docx parsing failed:",
        "transcription failed:",
        "failed to fetch url",
        "request failed:",
        "youtube transcript failed:",
        "failed to extract video id",
        "unsupported file type",
        "unsupported parser output format",
        "no readable text found",
    )
    return any(clean.startswith(marker) for marker in bad_markers)


# Minimum reranker confidence (Cohere's normalised relevance_score) required for a chunk to be
# considered "found in documents".  Scores below this trigger web fallback even
# when the vector store returns results, preventing irrelevant doc chunks from
# suppressing the web-search path for off-topic queries.
MIN_RELEVANCE_SCORE = 0.20


def is_small_talk(query: str) -> bool:
    """
    Strict heuristic gate for pure greetings / one-word social phrases.

    Intentionally narrow: only exact matches or very short prefix patterns so
    that substantive questions (e.g. "Hey, what is quantum computing?") are NOT
    classified as small-talk and still get document/web context.
    """
    normalized = re.sub(r"[^a-z0-9\s]", "", query.lower()).strip()
    if not normalized:
        return True

    # Only single-token or very short fixed phrases qualify
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
        "bye",
        "goodbye",
        "see you",
        "great",
        "nice",
        "cool",
    }
    if normalized in small_talk_exact:
        return True

    # Guard: only treat as small-talk if the full query is very short (≤5 words)
    # so that "Hey, tell me about machine learning" still goes through retrieval.
    word_count = len(normalized.split())
    if word_count > 5:
        return False

    small_talk_prefixes = (
        "hi ",
        "hello ",
        "hey ",
        "thanks ",
        "thank you ",
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


async def run_query(
    query: str,
    session_id: Optional[str] = None,
    top_k: int = 5,
    tts: bool = False,
    doc_ids: Optional[List[str]] = None,
) -> dict:
    """
    Core query/answer flow, independent of any web framework.  Returns a
    plain dict — callers (FastAPI route, Streamlit app) render it however
    they need to.
    """
    t0 = time.perf_counter()
    # Always produce a usable session ID; None is treated as "no session"
    resolved_session_id = session_id or str(uuid.uuid4())
    print(f"\n📨 Query: {query!r} | Session: {resolved_session_id}")

    fallback_used = False
    fallback_reason = None
    doc_context = ""
    context_source = "doc"   # tracks whether context came from docs or web
    final_chunks = []
    conversational = is_small_talk(query)

    # Resolve which doc IDs are in scope for this query
    allowed_doc_ids = set(doc_ids or [])
    if not allowed_doc_ids and session_id:  # only look up if caller sent a session
        allowed_doc_ids = set(session_docs.get_docs(session_id))
    has_doc_scope = len(allowed_doc_ids) > 0

    # Check whether the caller has session-scoped documents.
    # Even if no session docs are registered, we still check the FAISS
    # index — the user may have uploaded documents without a session.
    no_session_docs = (session_id is None) or (session_id and not has_doc_scope)

    if conversational:
        print("  💬 Small-talk intent detected → skipping retrieval/web fallback")
    elif no_session_docs:
        # No session docs, but the FAISS index may still have documents.
        # Check the index first before falling back to web search.
        vector_db.ensure_loaded()
        index_has_docs = (vector_db.index is not None and vector_db.index.ntotal > 0)
        if not index_has_docs:
            fallback_used = True
            fallback_reason = "no_session_docs"
            context_source = "web"
            print("  ⚠️  No documents registered and index is empty → web search")
            doc_context = await web_search.search_web(query)
        else:
            # Index has documents — treat as a global search (no doc_id scoping).
            print(f"  ℹ️  No session docs, but index has {vector_db.index.ntotal} vectors → searching index")
            allowed_doc_ids = set()  # clear any scope — search everything
            no_session_docs = False  # flag: will enter retrieval below

    if not conversational and not no_session_docs and not fallback_used and not doc_context:
        # ── Step 1: Check if the vector store has ANY documents ───────────
        vector_db.ensure_loaded()
        index_has_docs = (vector_db.index is not None and vector_db.index.ntotal > 0)
        print(f"  Index size: {vector_db.index.ntotal if vector_db.index else 0} vectors")

        if not index_has_docs:
            # No documents uploaded at all → use web search
            fallback_used = True
            context_source = "web"
            print("  ⚠️  Index is empty → web search")
            doc_context = await web_search.search_web(query)
        else:
            # ── Step 2: Retrieve top-k chunks ──────────────────────────────────
            try:
                if allowed_doc_ids:
                    # Directly retrieve chunks strictly scoped to the allowed docs.
                    # This avoids the sub-selection bug caused by post-filtering the global ANN search.
                    top_chunks = retrieve_scoped_chunks(
                        query,
                        allowed_doc_ids=allowed_doc_ids,
                        top_k=max(20, top_k * 10),
                    )
                    print(f"  Scoped search retrieved {len(top_chunks)} chunks for {len(allowed_doc_ids)} docs")
                else:
                    retrieval_k = max(10, top_k * 4)
                    top_chunks = retriever.hybrid_retrieve(query, top_k=retrieval_k)
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
                    max_rerank_candidates = min(len(raw_texts_clean), max(12, top_k * 4))
                    rerank_candidates = raw_texts_clean[:max_rerank_candidates]
                    reranked_texts, raw_scores, rerank_indices = _get_reranker().rerank(
                        query, rerank_candidates, top_n=top_k
                    )
                    # Cohere's rerank endpoint already returns a normalised [0,1]
                    # relevance_score — no sigmoid needed (that was only for the
                    # old local cross-encoder's raw logits).
                    norm_scores = [float(s) for s in raw_scores]
                    print(f"  Reranker scores: {[round(s, 3) for s in norm_scores]}")
                except Exception as e:
                    print(f"  Reranker error: {e} — using FAISS order")
                    reranked_texts = raw_texts_clean[:top_k]
                    norm_scores = [1.0] * len(reranked_texts)
                    rerank_indices = list(range(len(reranked_texts)))

                # ── Step 5: Map back to original chunk objects via indices ─
                for idx, norm_score in zip(rerank_indices, norm_scores):
                    if idx < len(top_chunks):
                        final_chunks.append(top_chunks[idx])
                        print(f"  🔹 Score={norm_score:.3f} | {raw_texts_clean[idx][:100]}")

                # ── Step 6: Relevance gate — score threshold check ────────
                # FAISS + reranker always return *something* from the index
                # even for completely unrelated queries.  We only accept doc
                # chunks when the best normalised reranker relevance score
                # clears MIN_RELEVANCE_SCORE (default 0.40).  Below that the
                # retrieved chunks are likely noise → fall back to web.
                best_score = max(norm_scores) if norm_scores else 0.0
                print(f"  Best reranker score: {best_score:.3f} (threshold={MIN_RELEVANCE_SCORE})")

                if best_score >= MIN_RELEVANCE_SCORE:
                    # Chunks are sufficiently relevant — build doc context.
                    doc_context = "\n\n".join(
                        strip_prefix(c["chunk"]) for c in final_chunks
                    )
                    fallback_used = False
                    context_source = "doc"
                else:
                    # Chunks exist but are not relevant to this query.
                    fallback_used = True
                    fallback_reason = "low_relevance"
                    context_source = "web"
                    print(
                        f"  ⚠️  Best score {best_score:.3f} < {MIN_RELEVANCE_SCORE} "
                        "→ chunks not relevant, falling back to web search"
                    )
                    doc_context = await web_search.search_web(query)
            else:
                if has_doc_scope:
                    # Session docs exist but retrieval returned nothing usable.
                    fallback_used = True
                    fallback_reason = "no_relevant_session_docs"
                    context_source = "web"
                    print("  ⚠️  No scoped document chunks found; falling back to web search")
                    doc_context = await web_search.search_web(query)
                else:
                    # FAISS returned 0 results (shouldn't happen if index_has_docs)
                    fallback_used = True
                    fallback_reason = "retrieval_empty"
                    context_source = "web"
                    print("  ⚠️  FAISS returned no results despite non-empty index → web search")
                    doc_context = await web_search.search_web(query)

    # ── Step 7: Build context and answer ─────────────────────────────
    combined_context = doc_context.strip()
    print(
        f"  📚 Context length: {len(combined_context)} chars | "
        f"fallback={fallback_used} | conversational={conversational}"
    )

    answer = await llm.answer_question(
        context=combined_context,
        question=query,
        session_id=resolved_session_id,
        use_documents=not conversational,
        # Tell the LLM where the context came from so it can frame its answer
        # correctly ("according to the document" vs "according to the web").
        context_source=context_source if not conversational else None,
    )

    if fallback_used and not conversational:
        if fallback_reason == "no_session_docs":
            prefix = "🌐 No uploaded document was provided for this session. Answering from the web."
        elif fallback_reason == "no_relevant_session_docs":
            prefix = "🌐 No relevant section was found in your uploaded documents. Answering from the web."
        elif fallback_reason == "low_relevance":
            prefix = "🌐 The documents don't seem to cover this topic. Answering from the web."
        else:
            prefix = "🌐 Document retrieval returned no usable context. Answering from the web."
        answer = f"{prefix}\n\n{answer}"

    print(f"  ⏱️ Query total: {time.perf_counter() - t0:.2f}s")
    print(f"  ✅ {answer[:120]}")

    # ── Step 8: Optional TTS ──────────────────────────────────────────
    tts_path = None
    tts_file_path = None
    if tts:
        try:
            tts_file_path = await generate_tts(answer)
            tts_path = f"/audio/{Path(tts_file_path).name}"
        except Exception as e:
            print(f"  TTS failed: {e}")

    return {
        "question": query,
        "session_id": resolved_session_id,
        "answer": answer,
        "top_chunks": final_chunks if not fallback_used else [],
        "fallback_used": fallback_used,
        "tts_audio_path": tts_path,
        # Absolute filesystem path, useful for callers (e.g. Streamlit) that
        # want to play the audio directly without going through an HTTP route.
        "tts_file_path": tts_file_path,
    }

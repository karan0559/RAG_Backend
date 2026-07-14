"""
Streamlit frontend for the Smart RAG System — an alternative interface to
app/main.py's FastAPI app, sharing the same app/Services/* backend.  Exists
so the app can be deployed free on Streamlit Community Cloud, which only
runs native Streamlit scripts (not arbitrary FastAPI/Docker apps).
"""
import os
import sys

# app/Services/*'s print() calls include emoji; Windows' default console
# codepage can't encode them and would crash mid-request otherwise. Matches
# the same defensive reconfigure app/main.py does for the FastAPI process.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import streamlit as st

st.set_page_config(page_title="Smart RAG System", page_icon="🧠", layout="wide")

# Streamlit Cloud's secrets manager populates st.secrets, not os.environ.
# Bridge them before importing any app.Services module, since several read
# os.getenv(...) at import time. Safe to run locally too: accessing
# st.secrets (even via `in`) raises StreamlitSecretNotFoundError, not just
# an empty mapping, when no .streamlit/secrets.toml exists at all — so the
# whole bridge is best-effort and each module's own load_dotenv() picks up
# .env as before if it's skipped.
try:
    for _key in ("GROQ_API_KEY", "GROQ_MODEL", "TAVILY_API_KEY", "COHERE_API_KEY"):
        if _key in st.secrets:
            os.environ[_key] = st.secrets[_key]
except Exception:
    pass

import asyncio
import re
import uuid
from pathlib import Path


def format_markdown(text: str) -> str:
    """
    LLM output uses '• ' as a bullet marker, but that character isn't valid
    CommonMark list syntax (only '-'/'*'/'+' are, and each item needs its
    own line) — without this it renders as one run-on paragraph. Give each
    bullet its own line with a real markdown list marker.
    """
    if not text or "•" not in text:
        return text
    return re.sub(r"\s*•\s*", "\n- ", text).strip()

from app.Services import retriever, llm
from app.Services.query_pipeline import run_query
from app.Services.upload_pipeline import (
    EXT_TO_TYPE,
    UPLOAD_DIR,
    process_uploaded_file,
    process_url_or_youtube,
)
from app.Memory import session_docs

CONVERTED_DIR = Path(__file__).resolve().parent / "data" / "converted_files"
CONVERTED_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_TYPES = [ext.lstrip(".") for ext in EXT_TO_TYPE]

# ── Session state ────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm your AI assistant. Upload files or paste a URL/YouTube link in "
            "the sidebar, then ask me anything! 🤖",
        }
    ]

session_id = st.session_state.session_id


# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 RAG Assistant")
    st.caption("Document Intelligence")

    st.subheader("Upload")
    uploaded_files = st.file_uploader(
        "Drop files or click to upload",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        help="PDF · DOCX · Images (OCR) · Audio (transcribed)",
    )
    if uploaded_files:
        for uf in uploaded_files:
            processed_key = f"processed_{uf.name}_{uf.size}"
            if processed_key in st.session_state:
                continue
            st.session_state[processed_key] = True

            ext = os.path.splitext(uf.name)[1].lower()
            if ext not in EXT_TO_TYPE:
                st.error(f"Unsupported file type: {uf.name}")
                continue

            save_path = UPLOAD_DIR / f"{uuid.uuid4()}{ext}"
            with open(save_path, "wb") as f:
                f.write(uf.getbuffer())

            with st.spinner(f"Processing {uf.name}..."):
                result = asyncio.run(
                    process_uploaded_file(str(save_path), uf.name, session_id=session_id)
                )

            if result["chunk_count"] > 0:
                st.success(f"✅ {uf.name} indexed ({result['chunk_count']} chunks)")
                if result.get("summary"):
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"📄 **{uf.name}**\n\n{format_markdown(result['summary'])}",
                        }
                    )
            else:
                st.warning(result.get("warning") or f"No usable text extracted from {uf.name}.")

    url_input = st.text_input("Paste URL or YouTube link...")
    if st.button("Go", use_container_width=True) and url_input:
        with st.spinner("Processing link..."):
            try:
                result = asyncio.run(process_url_or_youtube(url_input, session_id=session_id))
                if result["chunk_count"] > 0:
                    st.success(f"✅ Indexed ({result['chunk_count']} chunks)")
                    if result.get("summary"):
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": f"🔗 **{url_input}**\n\n{format_markdown(result['summary'])}",
                            }
                        )
                else:
                    st.warning("No usable content extracted from that link.")
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"URL processing failed: {e}")

    st.divider()
    st.subheader("Documents")
    doc_ids = session_docs.get_docs(session_id)
    if doc_ids:
        for d in doc_ids:
            st.caption(f"📄 {d}")
    else:
        st.caption("No documents yet.")

    col_new, col_tts = st.columns(2)
    with col_new:
        if st.button("New Chat", use_container_width=True):
            session_docs.clear_session(session_id)
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()
    with col_tts:
        tts_enabled = st.toggle("🔊 TTS")

    st.divider()
    st.subheader("Tools")

    with st.expander("🔄 Convert"):
        conv_file = st.file_uploader("File to convert", key="convert_uploader")
        output_format = st.selectbox("Convert to", ["docx", "pdf", "jpg", "xlsx"], key="convert_format")
        if conv_file and st.button("Convert", key="convert_btn"):
            file_ext = Path(conv_file.name).suffix.lower().strip(".")
            input_path = CONVERTED_DIR / f"{uuid.uuid4()}.{file_ext}"
            with open(input_path, "wb") as f:
                f.write(conv_file.getbuffer())

            output_filename = f"{Path(conv_file.name).stem}_converted.{output_format}"
            output_path = CONVERTED_DIR / output_filename

            try:
                if file_ext == "pdf" and output_format == "docx":
                    from app.Services.convertors.pdf_to_docx import convert_pdf_to_docx
                    convert_pdf_to_docx(str(input_path), str(output_path))
                elif file_ext in ("jpg", "jpeg", "png") and output_format == "pdf":
                    from app.Services.convertors.image_to_pdf import convert_image_to_pdf
                    convert_image_to_pdf(str(input_path), str(output_path))
                elif file_ext == "pdf" and output_format == "jpg":
                    from app.Services.convertors.pdf_to_jpg import convert_pdf_to_jpg
                    result_path = convert_pdf_to_jpg(str(input_path), str(CONVERTED_DIR))
                    output_path = Path(result_path)
                elif file_ext == "docx" and output_format == "pdf":
                    from app.Services.convertors.docx_to_pdf import convert_docx_to_pdf
                    convert_docx_to_pdf(str(input_path), str(output_path))
                elif file_ext == "txt" and output_format == "xlsx":
                    from app.Services.convertors.txt_to_excel import convert_txt_to_excel
                    convert_txt_to_excel(str(input_path), str(output_path))
                else:
                    raise ValueError(f"Unsupported conversion: {file_ext} → {output_format}")

                with open(output_path, "rb") as f:
                    st.download_button(
                        "Download converted file",
                        f.read(),
                        file_name=Path(output_path).name,
                        key="convert_download",
                    )
            except Exception as e:
                st.error(f"Conversion failed: {e}")

    with st.expander("📊 Compare / Summarize"):
        st.caption("Add documents here if they aren't already in the list below.")
        compare_uploads = st.file_uploader(
            "Upload documents for comparison",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            key="compare_uploader",
        )
        if compare_uploads and st.button("Add to documents", key="compare_add_btn"):
            any_processed = False
            for cu in compare_uploads:
                ext = os.path.splitext(cu.name)[1].lower()
                if ext not in EXT_TO_TYPE:
                    st.error(f"Unsupported file type: {cu.name}")
                    continue
                save_path = UPLOAD_DIR / f"{uuid.uuid4()}{ext}"
                with open(save_path, "wb") as f:
                    f.write(cu.getbuffer())
                with st.spinner(f"Processing {cu.name}..."):
                    asyncio.run(process_uploaded_file(str(save_path), cu.name, session_id=session_id))
                any_processed = True
            if any_processed:
                st.rerun()

        selected_docs = st.multiselect("Documents", doc_ids, key="compare_docs")
        compare_mode = st.radio("Mode", ["compare", "summarize"], key="compare_mode", horizontal=True)
        compare_question = st.text_input("Question (compare mode only)", key="compare_question")
        if st.button("Run", key="compare_btn") and selected_docs:
            doc_chunks = retriever.get_chunks_by_doc_ids(selected_docs)
            if not doc_chunks:
                st.error("No matching documents found.")
            else:
                with st.spinner("Working..."):
                    if compare_mode == "summarize":
                        all_text = "\n\n".join(["\n".join(chunks) for chunks in doc_chunks.values()])
                        result_text = asyncio.run(llm.summarize_text(all_text))
                    else:
                        result_text = asyncio.run(llm.compare_documents(doc_chunks, question=compare_question))
                st.markdown(format_markdown(result_text))


# ── Main chat area ───────────────────────────────────────────────────────
st.title("Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(format_markdown(msg["content"]))
        if msg.get("audio_path") and os.path.exists(msg["audio_path"]):
            st.audio(msg["audio_path"])

if prompt := st.chat_input("Ask anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = asyncio.run(
                run_query(query=prompt, session_id=session_id, top_k=5, tts=tts_enabled, doc_ids=None)
            )
        st.markdown(format_markdown(result["answer"]))
        audio_path = result.get("tts_file_path")
        if audio_path:
            st.audio(audio_path)

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "audio_path": result.get("tts_file_path")}
    )

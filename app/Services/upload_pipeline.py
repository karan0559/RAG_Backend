"""
Framework-agnostic core of document ingestion: extract, embed, register to
a session, and auto-summarise.

Extracted out of app/Routes/upload.py so both the FastAPI route and the
Streamlit app can call the exact same logic on a plain filesystem path
instead of a FastAPI-specific UploadFile.
"""
import os
from pathlib import Path
from typing import Optional

from app.Services import extractor, embedder, llm
from app.Memory import session_docs

# Absolute path — safe regardless of the directory the app is launched from.
UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

EXT_TO_TYPE = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".mp3": "audio",
    ".wav": "audio",
    ".ogg": "audio",
    ".m4a": "audio",
    ".webm": "audio",
}

# How many chunks / chars to feed the summariser.
# Keeps the prompt well within the LLM context window.
_SUMMARY_MAX_CHUNKS = 30
_SUMMARY_MAX_CHARS = 6000


async def summarise_chunks(chunks: list, label: str) -> Optional[str]:
    """
    Generate a plain-language summary from the first N chunks of a document.

    Returns None silently on failure so the upload still succeeds even if
    the LLM is unavailable or rate-limited.

    The summary includes:
      - A short overview of what the file contains (3-5 sentences)
      - 3-5 example questions the user can ask, as a bullet list

    This is displayed in the chat right after upload so the user immediately
    knows what's in the file without having to guess search terms.
    """
    if not chunks:
        return None

    sample_text = "\n\n".join(chunks[:_SUMMARY_MAX_CHUNKS])[:_SUMMARY_MAX_CHARS]
    try:
        payload = {
            "model": llm.GROQ_MODEL,
            "temperature": 0.4,
            "max_tokens": 512,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful document assistant. "
                        "When given the content of a file, provide a concise summary (3-5 sentences) "
                        "covering what the file is about and its main topics. "
                        "Then suggest 3-5 specific questions the user could ask about this content. "
                        "Format the questions as a bullet list starting with •"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"File: {label}\n\nContent:\n{sample_text}\n\n"
                        "Please summarise this and suggest questions I can ask."
                    ),
                },
            ],
        }
        return await llm.call_groq_llm(payload)
    except Exception as e:
        print(f"  ⚠️  Auto-summary failed for '{label}': {e}")
        return None


async def process_uploaded_file(save_path: str, filename: str, session_id: Optional[str] = None) -> dict:
    """
    Extract, embed, and auto-summarise a file that has already been saved
    to disk at `save_path`.  `filename` is the original filename (used for
    extension detection, doc_id derivation, and display).
    """
    ext = os.path.splitext(filename)[1].lower()
    file_type = EXT_TO_TYPE.get(ext)
    if not file_type:
        raise ValueError(f"Unsupported file extension: {ext}")

    print(f"  📁 Saved: {save_path} ({file_type})")

    chunks = extractor.extract_content(str(save_path), file_type)
    doc_id = filename.rsplit(".", 1)[0]
    extraction_warning = None
    summary = None

    if isinstance(chunks, list) and chunks:
        embedder.embed_chunks(chunks, doc_id=doc_id)
        if session_id:
            session_docs.register_doc(session_id=session_id, doc_id=doc_id)
        print(f"  ✅ Embedded {len(chunks)} chunks for '{doc_id}'")

        # Auto-summarise immediately so the user knows what's in the file
        # and can ask relevant questions without guessing keywords.
        summary = await summarise_chunks(chunks, label=filename)
        if summary:
            print(f"  📝 Auto-summary generated for '{doc_id}'")
    else:
        chunks = []
        extraction_warning = "No usable text extracted from file; nothing was indexed."
        print(f"  ⚠️  No valid chunks for '{doc_id}'")

    return {
        "session_id": session_id,
        "doc_id": doc_id,
        "original_filename": filename,
        "file_type": file_type,
        "saved_path": str(save_path),
        "chunk_count": len(chunks),
        "summary": summary,          # None if extraction failed or LLM unavailable
        "warning": extraction_warning,
    }


async def process_url_or_youtube(input_text: str, session_id: Optional[str] = None) -> dict:
    """Extract, embed, and auto-summarise a URL or YouTube link."""
    input_text = input_text.strip()
    if "youtube.com" in input_text or "youtu.be" in input_text:
        file_type = "youtube"
    elif input_text.startswith("http"):
        file_type = "url"
    else:
        raise ValueError("Invalid or unsupported input_text format.")

    chunks = extractor.extract_content(input_text, file_type)
    doc_id = input_text
    summary = None

    if isinstance(chunks, list) and chunks:
        embedder.embed_chunks(chunks, doc_id=doc_id)
        if session_id:
            session_docs.register_doc(session_id=session_id, doc_id=doc_id)
        summary = await summarise_chunks(chunks, label=input_text)

    return {
        "doc_id": doc_id,
        "original_filename": input_text,
        "file_type": file_type,
        "saved_path": "N/A",
        "chunk_count": len(chunks) if isinstance(chunks, list) else 1,
        "summary": summary,
    }

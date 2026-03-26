from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional, Union
from pathlib import Path
import os
import uuid

from app.Services import extractor
from app.Services import embedder
from app.Memory import session_docs

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter()

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


async def _process_single_file(file: UploadFile, session_id: Optional[str] = None) -> dict:
    """Save, extract, embed a single uploaded file. Returns result dict."""
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    file_type = EXT_TO_TYPE.get(ext)

    if not file_type:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {ext}")

    uid = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{uid}{ext}"

    with open(save_path, "wb") as f:
        f.write(await file.read())

    print(f"  📁 Saved: {save_path} ({file_type})")

    chunks = extractor.extract_content(str(save_path), file_type)
    doc_id = filename.rsplit(".", 1)[0]

    if isinstance(chunks, list) and chunks:
        embeddings = embedder.embed_chunks(chunks, doc_id=doc_id)
        if session_id:
            session_docs.register_doc(session_id=session_id, doc_id=doc_id)
        print(f"  ✅ Embedded {len(chunks)} chunks for '{doc_id}'")
    else:
        chunks = [chunks] if isinstance(chunks, str) else []
        embeddings = []
        print(f"  ⚠️  No valid chunks for '{doc_id}'")

    return {
        "id": uid,
        "doc_id": doc_id,
        "original_filename": filename,
        "file_type": file_type,
        "saved_path": str(save_path),
        "chunk_count": len(chunks),
        "embedding_shape": str(getattr(embeddings, "shape", f"{len(embeddings)} x dim")),
        "text_preview": chunks[:3],
    }


@router.post("/", summary="Upload one or more files, or a URL/YouTube link")
async def upload_file_or_input(
    file: Optional[UploadFile] = File(default=None),
    files: Optional[List[UploadFile]] = File(default=None),
    input_text: Union[str, None] = Form(default=None),
    session_id: Optional[str] = Form(default=None),
):
    """
    Accepts:
    - A single file under the key 'file'
    - Multiple files under the key 'files'
    - A URL or YouTube link under the key 'input_text'
    """

    # ── URL / YouTube input ────────────────────────────────────────────────
    if input_text:
        input_text = input_text.strip()
        if "youtube.com" in input_text or "youtu.be" in input_text:
            file_type = "youtube"
        elif input_text.startswith("http"):
            file_type = "url"
        else:
            raise HTTPException(status_code=400, detail="Invalid or unsupported input_text format.")

        try:
            chunks = extractor.extract_content(input_text, file_type)
            doc_id = input_text
            uid = str(uuid.uuid4())

            if isinstance(chunks, list) and chunks:
                embedder.embed_chunks(chunks, doc_id=doc_id)
                if session_id:
                    session_docs.register_doc(session_id=session_id, doc_id=doc_id)

            return JSONResponse({
                "id": uid,
                "doc_id": doc_id,
                "original_filename": input_text,
                "file_type": file_type,
                "saved_path": "N/A",
                "chunk_count": len(chunks) if isinstance(chunks, list) else 1,
                "text_preview": chunks[:3] if isinstance(chunks, list) else [chunks],
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"URL processing failed: {e}")

    # ── File upload(s) ─────────────────────────────────────────────────────
    # Normalise: accept both 'file' (single) and 'files' (multiple) keys
    all_files: List[UploadFile] = []
    if files:
        all_files.extend(files)
    if file:
        all_files.append(file)

    if not all_files:
        raise HTTPException(status_code=400, detail="Please upload a file or provide a URL.")

    # Single file → return a simple dict for backwards compatibility
    if len(all_files) == 1:
        try:
            result = await _process_single_file(all_files[0], session_id=session_id)
            return JSONResponse(result)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    # Multiple files → process each and return a list
    results = []
    errors = []
    for f in all_files:
        try:
            r = await _process_single_file(f, session_id=session_id)
            results.append(r)
        except Exception as e:
            errors.append({"filename": f.filename, "error": str(e)})

    return JSONResponse({
        "files_processed": len(results),
        "files_failed": len(errors),
        "results": results,
        "errors": errors,
        # Aggregate stats
        "chunks_created": sum(r["chunk_count"] for r in results),
    })

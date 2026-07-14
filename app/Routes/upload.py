from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional, Union
import os
import uuid

from app.Services.upload_pipeline import (
    UPLOAD_DIR,
    EXT_TO_TYPE,
    process_uploaded_file,
    process_url_or_youtube,
)

router = APIRouter()


async def _save_and_process(file: UploadFile, session_id: Optional[str] = None) -> dict:
    """Save an uploaded FastAPI file to disk, then run the shared pipeline."""
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in EXT_TO_TYPE:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {ext}")

    uid = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{uid}{ext}"
    with open(save_path, "wb") as f:
        f.write(await file.read())

    result = await process_uploaded_file(str(save_path), filename, session_id=session_id)
    return {"id": uid, **result}


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
        try:
            result = await process_url_or_youtube(input_text, session_id=session_id)
            return JSONResponse({"id": str(uuid.uuid4()), **result})
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"URL processing failed: {e}")

    # ── File upload(s) ─────────────────────────────────────────────────────
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
            result = await _save_and_process(all_files[0], session_id=session_id)
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
            r = await _save_and_process(f, session_id=session_id)
            results.append(r)
        except Exception as e:
            errors.append({"filename": f.filename, "error": str(e)})

    return JSONResponse({
        "files_processed": len(results),
        "files_failed": len(errors),
        "results": results,
        "errors": errors,
        "chunks_created": sum(r["chunk_count"] for r in results),
    })

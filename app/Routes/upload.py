from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Union, Optional
from pathlib import Path
import os
import uuid

from app.Services import extractor
from app.Services import embedder

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter()

@router.post("/upload/", summary="Upload file or input URL/YouTube link")
async def upload_file_or_input(
    file: Optional[UploadFile] = File(default=None),
    input_text: Union[str, None] = Form(default=None)
):
    if file and input_text:
        raise HTTPException(status_code=400, detail="Please provide either a file or a URL/YouTube link, not both.")

    ext_to_type = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".jpg": "image",
        ".jpeg": "image",
        ".png": "image",
        ".mp3": "audio",
        ".wav": "audio",
        ".ogg": "audio",
        ".m4a": "audio",
        ".webm": "audio"
    }

    try:
        # FILE UPLOAD PATH
        if file:
            filename = file.filename
            ext = os.path.splitext(filename)[1].lower()
            file_type = ext_to_type.get(ext)

            if not file_type:
                raise HTTPException(status_code=400, detail=f"Unsupported file extension: {ext}")

            uid = str(uuid.uuid4())
            save_path = UPLOAD_DIR / f"{uid}{ext}"

            with open(save_path, "wb") as f:
                f.write(await file.read())

            print(f"📦 File saved at: {save_path}")
            print(f"📂 File type detected: {file_type}")

            chunks = extractor.extract_content(str(save_path), file_type)
            doc_id = filename.rsplit(".", 1)[0]

        # TEXT INPUT PATH 
        elif input_text:
            input_text = input_text.strip()
            if "youtube.com" in input_text or "youtu.be" in input_text:
                file_type = "youtube"
            elif input_text.startswith("http"):
                file_type = "url"
            else:
                raise HTTPException(status_code=400, detail="Invalid or unsupported input_text format.")

            chunks = extractor.extract_content(input_text, file_type)
            doc_id = input_text
            filename = input_text
            save_path = "N/A"
            uid = str(uuid.uuid4())

        else:
            raise HTTPException(status_code=400, detail="Please upload a file or enter a valid input_text.")

        # Embedding chunks with doc_id
        if isinstance(chunks, list) and chunks:
            embeddings = embedder.embed_chunks(chunks, doc_id=doc_id)
            print("✅ Embedding complete.")
        else:
            chunks = [chunks] if isinstance(chunks, str) else []
            embeddings = []
            print("⚠️ No valid chunks to embed.")

        return JSONResponse({
            "id": uid,
            "doc_id": doc_id,  
            "original_filename": filename,
            "file_type": file_type,
            "saved_path": str(save_path),
            "chunk_count": len(chunks),
            "embedding_shape": str(getattr(embeddings, 'shape', f'{len(embeddings)} x dim')),
            "text_preview": chunks[:3]
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

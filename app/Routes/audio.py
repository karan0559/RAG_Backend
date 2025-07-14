from fastapi import APIRouter
from fastapi.responses import FileResponse
import os

router = APIRouter()

AUDIO_DIR = "data/audio_responses"

@router.get("/{filename}", summary="Serve TTS audio")
async def get_audio(filename: str):
    filepath = os.path.join(AUDIO_DIR, filename)
    print(f"🎧 Serving audio: {filepath}")
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="audio/mpeg", filename=filename)
    else:
        return {"error": "File not found"}

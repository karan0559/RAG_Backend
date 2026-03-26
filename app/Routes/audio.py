from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

router = APIRouter()

# Absolute path — safe regardless of the directory uvicorn is launched from.
AUDIO_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "audio_responses"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/{filename}", summary="Serve TTS audio")
async def get_audio(filename: str):
    filepath = AUDIO_DIR / filename
    print(f"[Audio] Serving: {filepath}")
    if filepath.exists():
        return FileResponse(str(filepath), media_type="audio/mpeg", filename=filename)
    # Bug fix: was returning 200 OK with an error dict, which caused the browser
    # to try to play the error response as audio.
    raise HTTPException(status_code=404, detail=f"Audio file not found: {filename}")

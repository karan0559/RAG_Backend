import os
import mimetypes
import httpx
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"


def transcribe_audio(path: str) -> list[str]:
    """
    Transcribe an audio file via Groq's hosted Whisper endpoint.
    Groq accepts mp3/wav/ogg/m4a/webm/flac/mp4/mpeg/mpga directly, so no
    local ffmpeg conversion step is needed.
    """
    try:
        if not GROQ_API_KEY:
            return ["Transcription failed: GROQ_API_KEY is not set in .env"]

        content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        print(f"[AudioTranscriber] Transcribing via Groq: {path}")

        with open(path, "rb") as f:
            response = httpx.post(
                GROQ_TRANSCRIPTION_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": (os.path.basename(path), f, content_type)},
                data={"model": GROQ_WHISPER_MODEL, "response_format": "json"},
                timeout=60.0,
            )

        if response.status_code != 200:
            raise RuntimeError(f"Groq transcription failed: {response.status_code} - {response.text}")

        text = response.json().get("text", "").strip()
        return [text] if text else []
    except Exception as e:
        return [f"Transcription failed: {e}"]

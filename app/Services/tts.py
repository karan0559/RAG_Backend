import asyncio
from edge_tts import Communicate
from uuid import uuid4
from pathlib import Path

# Absolute path — safe regardless of the directory uvicorn is launched from.
TTS_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "audio_responses"
TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


async def generate_tts(text: str) -> str:
    """Generate TTS audio and return the absolute file path as a string."""
    filename = f"{uuid4().hex}.mp3"
    filepath = TTS_OUTPUT_DIR / filename

    communicate = Communicate(text=text, voice="en-US-JennyNeural")
    await communicate.save(str(filepath))

    return str(filepath)

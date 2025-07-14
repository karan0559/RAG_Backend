import os
import asyncio
from edge_tts import Communicate
from uuid import uuid4

TTS_OUTPUT_DIR = "data/audio_responses"
os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)

async def generate_tts(text: str) -> str:
    filename = f"{uuid4().hex}.mp3"
    filepath = os.path.join(TTS_OUTPUT_DIR, filename)
    
    communicate = Communicate(text=text, voice="en-US-JennyNeural")
    await communicate.save(filepath)
    
    return filepath

import os
import subprocess
import shutil
from faster_whisper import WhisperModel

model = WhisperModel("base", device="cpu")

if shutil.which("ffmpeg") is None:
    raise EnvironmentError("❌ FFmpeg not found in system PATH. Please install and add it to PATH.")

def convert_to_wav(input_path: str, target_sr=16000) -> str:
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", str(target_sr),  
        "-ac", "1",            
        output_path
    ]
    print(f" Converting {input_path} → {output_path}")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

def transcribe_audio(path: str):
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext != ".wav":
            path = convert_to_wav(path)

        print(f" Transcribing file: {path}")
        segments, _ = model.transcribe(path)

        return [segment.text for segment in segments if segment.text.strip()]
    except Exception as e:
        return [f" Transcription failed: {e}"]

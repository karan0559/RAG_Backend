import os
import subprocess
import shutil

# Module-level sentinel — model is loaded on first use, not at import time.
_whisper_model = None


def _get_whisper_model():
    """Lazy-load the Whisper model once and cache it."""
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
            print("[AudioTranscriber] Loading Whisper model (base, cpu)…")
            _whisper_model = WhisperModel("base", device="cpu")
            print("[AudioTranscriber] Whisper model ready.")
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}") from e
    return _whisper_model


def _check_ffmpeg():
    """Raise a clear error if FFmpeg is not available."""
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError(
            "FFmpeg not found in system PATH. "
            "Install FFmpeg and make sure it is on PATH before transcribing audio."
        )


def convert_to_wav(input_path: str, target_sr: int = 16000) -> str:
    _check_ffmpeg()
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", str(target_sr),
        "-ac", "1",
        output_path,
    ]
    print(f"[AudioTranscriber] Converting {input_path} → {output_path}")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


def transcribe_audio(path: str) -> list[str]:
    try:
        _check_ffmpeg()
        ext = os.path.splitext(path)[1].lower()
        if ext != ".wav":
            path = convert_to_wav(path)

        print(f"[AudioTranscriber] Transcribing: {path}")
        model = _get_whisper_model()
        segments, _ = model.transcribe(path)
        return [seg.text for seg in segments if seg.text.strip()]
    except Exception as e:
        return [f"Transcription failed: {e}"]

"""
Voice handler — Speech-to-Text using OpenAI Whisper (local, free).
Requires: pip install openai-whisper
"""

import os
import tempfile
from pathlib import Path

try:
    import whisper
    _model = whisper.load_model("base")   # tiny | base | small | medium | large
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    _model = None


def transcribe_audio(audio_bytes: bytes, ext: str = "webm") -> str:
    """
    Transcribe raw audio bytes to text.
    Returns the transcribed string, or raises RuntimeError if Whisper is unavailable.
    """
    if not WHISPER_AVAILABLE or _model is None:
        raise RuntimeError(
            "Whisper is not installed. Run: pip install openai-whisper"
        )

    suffix = f".{ext.lstrip('.')}"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = _model.transcribe(tmp_path)
        return result.get("text", "").strip()
    finally:
        os.unlink(tmp_path)
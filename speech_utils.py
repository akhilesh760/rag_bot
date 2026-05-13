"""Speech-to-text (Groq Whisper) and text-to-speech (gTTS) helpers."""

import base64
import io
import os

from gtts import gTTS

from rag_pipeline import client

GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"
TTS_LANG = "en"
TTS_MAX_CHARS = 5000


def transcribe_audio(audio_path: str) -> str:
    """
    Convert speech file to plain text via Groq's Whisper-compatible API.
    """
    filename = (
        os.path.basename(audio_path)
        or "recording.wav"
    )

    with open(audio_path, "rb") as f:

        payload = f.read()

    upload = io.BytesIO(payload)

    upload.name = filename

    transcription = client.audio.transcriptions.create(
        file=upload,
        model=GROQ_WHISPER_MODEL,
    )

    text = getattr(transcription, "text", None) or ""

    return text.strip()


def text_to_speech_base64(text: str) -> dict:
    """
    Synthesize spoken audio (MP3) and return base64 payload for JSON transport.
    """
    clipped = (
        text[:TTS_MAX_CHARS]
        if len(text) > TTS_MAX_CHARS
        else text
    )

    buf = io.BytesIO()

    tts = gTTS(text=clipped, lang=TTS_LANG)

    tts.write_to_fp(buf)

    raw = buf.getvalue()

    return {
        "mime": "audio/mpeg",
        "base64": base64.standard_b64encode(raw).decode("ascii"),
    }

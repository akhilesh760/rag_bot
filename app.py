import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from rag_pipeline import ask_question
from speech_utils import text_to_speech_base64, transcribe_audio

AUDIO_DIR = Path(__file__).resolve().parent / "audio"


@asynccontextmanager
async def lifespan(_app: FastAPI):

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    yield


class ChatRequest(BaseModel):

    query: str


class ChatResponse(BaseModel):

    response: str
    status: str = "success"


app = FastAPI(
    title="RAG Voice Chat API",
    lifespan=lifespan,
)


@app.get("/health")
async def health():

    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):

    answer = ask_question(req.query)

    return ChatResponse(response=answer)


@app.post("/voice-chat")
async def voice_chat(
    audio: UploadFile = File(...)
):

    try:

        original = audio.filename or "recording.wav"

        suffix = Path(original).suffix.lower()

        allowed = {
            ".wav",
            ".webm",
            ".mp3",
            ".m4a",
            ".oga",
            ".ogg",
            ".flac",
        }

        if suffix not in allowed:

            suffix = ".wav"

        safe_name = f"{uuid.uuid4().hex}{suffix}"

        audio_path = AUDIO_DIR / safe_name

        content = await audio.read()

        audio_path.write_bytes(content)

        print(f"Saved Audio: {audio_path}")

        user_query = transcribe_audio(str(audio_path))

        print(f"Recognized Query: {user_query}")

        if not user_query:

            return {
                "status": "error",
                "message": "No speech detected",
                "query": "",
                "response": "",
                "response_audio_base64": "",
                "response_audio_mime": "",
            }

        chatbot_response = ask_question(user_query)

        tts = text_to_speech_base64(chatbot_response)

        return {
            "status": "success",
            "query": user_query,
            "response": chatbot_response,
            "response_audio_base64": tts["base64"],
            "response_audio_mime": tts["mime"],
        }

    except Exception as e:

        print(f"Voice Chat Error: {e}")

        return {
            "status": "error",
            "message": str(e),
            "query": "",
            "response": "",
            "response_audio_base64": "",
            "response_audio_mime": "",
        }

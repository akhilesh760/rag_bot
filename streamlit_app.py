import base64
import time

import requests
import streamlit as st

from streamlit_mic_recorder import mic_recorder

# -----------------------------------
# Page Config
# -----------------------------------

st.set_page_config(
    page_title="RAG Chat",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------
# Custom Styling
# -----------------------------------

st.markdown(
    """
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

h1, h2, h3 {
    letter-spacing: -0.02em;
}

div[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 0.25rem 0.75rem;
}

</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------
# Sidebar
# -----------------------------------

with st.sidebar:

    st.markdown("## RAG Chat")

    st.caption(
        "Voice-enabled RAG Assistant"
    )

    api_base = st.text_input(
        "API base URL",
        value="http://127.0.0.1:8000"
    )

    show_raw = st.toggle(
        "Show raw response JSON",
        value=False
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "Clear chat",
            use_container_width=True
        ):

            st.session_state.messages = []

            st.rerun()

    with col2:

        st.link_button(
            "Health check",
            f"{api_base}/health",
            use_container_width=True
        )

# -----------------------------------
# Main Header
# -----------------------------------

st.markdown("## Assistant")

st.caption(
    "Ask questions using text or voice."
)

# -----------------------------------
# Chat History
# -----------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content":
            "Hi! Ask me anything about your knowledge base."
        }
    ]

# -----------------------------------
# Display Messages
# -----------------------------------

for m in st.session_state.messages:

    with st.chat_message(m["role"]):

        st.markdown(m["content"])

# -----------------------------------
# Resolve user turn + answer (voice or text)
# -----------------------------------

user_text = None
answer_text = None
raw_backend = None
voice_audio_bytes = None

# -----------------------------------
# Voice Recorder
# -----------------------------------

st.markdown("### 🎤 Voice Input")

audio = mic_recorder(
    start_prompt="Start Recording",
    stop_prompt="Stop Recording",
    just_once=True,
    use_container_width=True
)

if audio:

    files = {
        "audio": (
            "voice.wav",
            audio["bytes"],
            "audio/wav"
        )
    }

    try:

        with st.spinner(
            "Transcribing audio and querying the assistant…"
        ):

            response = requests.post(
                f"{api_base}/voice-chat",
                files=files,
                timeout=120
            )

            data = response.json()

        st.success(
            "Voice recorded successfully!"
        )

        if data.get("status") == "success":

            user_text = (
                data.get("query")
                or ""
            ).strip()

            answer_text = (
                data.get("response")
                or ""
            ).strip()

            raw_backend = data

            b64 = data.get(
                "response_audio_base64"
            )

            if b64:

                voice_audio_bytes = base64.standard_b64decode(
                    b64
                )

        else:

            st.error(
                data.get(
                    "message",
                    "Voice processing failed.",
                )
            )

            if show_raw:

                st.json(data)

    except Exception as e:

        st.error(f"Voice Error: {e}")

# -----------------------------------
# Text Input
# -----------------------------------

text_prompt = st.chat_input(
    "Type your question…"
)

if (
    user_text is None
    and text_prompt
):

    user_text = text_prompt.strip()

    if user_text:

        try:

            with st.spinner("Thinking…"):

                resp = requests.post(
                    f"{api_base}/chat",
                    json={"query": user_text},
                    timeout=60,
                )

                raw_backend = resp.json()

            answer_text = (
                raw_backend.get("response")
                or raw_backend.get("message")
                or ""
            ).strip()

            if not answer_text:

                answer_text = "No response."

        except Exception as e:

            raw_backend = {
                "status": "error",
                "message": str(e)
            }

            answer_text = (
                f"Backend error: {e}"
            )

if user_text:

    # Skip empty transcripts
    if not user_text.strip():

        st.warning(
            "No text recognized from voice."
        )

    elif answer_text is not None:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        with st.chat_message("user"):

            st.markdown(user_text)

        with st.chat_message("assistant"):

            placeholder = st.empty()

            out = ""

            for ch in answer_text:

                out += ch

                placeholder.markdown(out)

                time.sleep(0.0015)

            if voice_audio_bytes is not None:

                st.audio(
                    voice_audio_bytes,
                    format="audio/mpeg",
                )

            if (
                show_raw
                and raw_backend is not None
            ):

                st.divider()

                st.json(raw_backend)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer_text,
            }
        )
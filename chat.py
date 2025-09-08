# streamlit_chat_text_or_voice.py
# streamlit_chat_text_or_voice_live.py
import streamlit as st
from langchain_groq import ChatGroq
from litellm import completion
from dotenv import load_dotenv
import os
import json
import numpy as np

# ==========================
# LLM Wrapper
# ==========================
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

class GroqLLM(ChatGroq):
    def __init__(self, **kwargs):
        super().__init__(model="groq/llama-3.1-8b-instant", **kwargs)

    def _call(self, prompt: str, stop=None) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = completion(model="groq/llama-3.1-8b-instant", messages=messages)
        return response["choices"][0]["message"]["content"]

llm = GroqLLM()

# ==========================
# Conversation Memory
# ==========================
if "memory" not in st.session_state:
    st.session_state.memory = []

def run_conversation(user_input):
    st.session_state.memory.append({"role": "user", "content": user_input})
    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.memory])
    output = llm._call(prompt)
    st.session_state.memory.append({"role": "assistant", "content": output})
    return output

# ==========================
# Vosk Model Setup
# ==========================
import wget
import zipfile
from vosk import Model, KaldiRecognizer

if not os.path.exists("model"):
    url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    zip_path = "vosk_model.zip"
    wget.download(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")
    os.rename("vosk-model-small-en-us-0.15", "model")

model = Model("model")

def speech_to_text_vosk(frames, sample_rate=16000):
    recognizer = KaldiRecognizer(model, sample_rate)
    recognizer.SetWords(True)
    audio_bytes = frames.tobytes()
    if recognizer.AcceptWaveform(audio_bytes):
        result = json.loads(recognizer.Result())
        return result.get("text", "")
    else:
        partial = json.loads(recognizer.PartialResult())
        return partial.get("partial", "")

# ==========================
# Streamlit WebRTC for Live Voice
# ==========================
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av

class VoskAudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.frames = []

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten()
        audio_int16 = (audio * 32767).astype(np.int16)
        self.frames.append(audio_int16)
        return frame

def get_live_transcription():
    ctx = webrtc_streamer(
        key="live-voice",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=VoskAudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )
    text = None
    if ctx.audio_processor and ctx.audio_processor.frames:
        frames = np.concatenate(ctx.audio_processor.frames)
        text = speech_to_text_vosk(frames)
        ctx.audio_processor.frames = []  # clear after processing
    return text

# ==========================
# Streamlit UI
# ==========================
st.title("🛎️ Customer Service Agent")

mode = st.radio("Choose mode:", ["Text", "Voice"])

# --------------------------
# Text Mode
# --------------------------
if mode == "Text":
    user_input = st.text_input("Type your message:")
    if st.button("Send") and user_input:
        response = run_conversation(user_input)
        st.text_area(
            "Conversation",
            value="\n".join([f"You: {m['content']}" if m['role'] == 'user' else f"AI: {m['content']}" for m in st.session_state.memory]),
            height=300
        )

# --------------------------
# Voice Mode
# --------------------------
# --------------------------
# Voice Mode with Start + Transcribe
# --------------------------
elif mode == "Voice":
    st.write("🎙️ Voice Mode")

    if "recording_started" not in st.session_state:
        st.session_state.recording_started = False

    # Start button
    if st.button("Start Listening"):
        st.session_state.recording_started = True
        st.info("🎧 Listening... Speak now!")

    # Show WebRTC player only if recording started
    if st.session_state.recording_started:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
        import av
        import numpy as np

        class VoskAudioProcessor(AudioProcessorBase):
            def __init__(self) -> None:
                self.frames = []

            def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
                audio = frame.to_ndarray().flatten()
                audio_int16 = (audio * 32767).astype(np.int16)
                self.frames.append(audio_int16)
                return frame

        if "webrtc_ctx" not in st.session_state:
            st.session_state.webrtc_ctx = webrtc_streamer(
                key="live-voice",
                mode=WebRtcMode.SENDONLY,
                audio_processor_factory=VoskAudioProcessor,
                media_stream_constraints={"audio": True, "video": False},
                async_processing=True,
            )

        # Transcribe button
        if st.button("Send / Transcribe"):
            ctx = st.session_state.webrtc_ctx
            if ctx.audio_processor and ctx.audio_processor.frames:
                frames = np.concatenate(ctx.audio_processor.frames)
                from vosk import Model, KaldiRecognizer
                import json

                recognizer = KaldiRecognizer(model, 16000)
                recognizer.SetWords(True)
                text = ""
                audio_bytes = frames.tobytes()
                if recognizer.AcceptWaveform(audio_bytes):
                    text = json.loads(recognizer.Result()).get("text", "")
                else:
                    text = json.loads(recognizer.PartialResult()).get("partial", "")

                st.success(f"You said: {text}")
                if text:
                    response = run_conversation(text)
                    st.text_area(
                        "Conversation",
                        value="\n".join([f"You: {m['content']}" if m['role'] == 'user' else f"AI: {m['content']}" for m in st.session_state.memory]),
                        height=300
                    )
                else:
                    st.warning("⚠️ Could not capture speech. Try again.")
                ctx.audio_processor.frames = []  # clear after processing

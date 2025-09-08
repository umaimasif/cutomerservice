# streamlit_chat_text_or_voice.py
import streamlit as st
import speech_recognition as sr
from langchain_groq import ChatGroq
from litellm import completion
from langchain.schema import HumanMessage

# ==========================
# LLM Wrapper
# ==========================
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

class GroqLLM(ChatGroq):
    def __init__(self, **kwargs):
        super().__init__(model="groq\llama-3.1-8b-instant", **kwargs)

    def _call(self, prompt: str, stop=None) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = completion(model="groq\llama-3.1-8b-instant", messages=messages)
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
# Voice Input Function
# ==========================
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import speech_recognition as sr
import av
import numpy as np

st.title("🎙️ Live Speech-to-Text")

recognizer = sr.Recognizer()

def speech_to_text():
    class STTProcessor(AudioProcessorBase):
        def __init__(self) -> None:
            self.frames = []

        def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
            # Convert frame to numpy (float32 PCM), then scale to int16
            audio = frame.to_ndarray().flatten()
            audio_int16 = (audio * 32767).astype("int16")
            self.frames.append(audio_int16)
            return frame

    ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=STTProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )

    text = None
    if ctx.audio_processor:
        if st.button("🎙️ Transcribe now"):
            if ctx.audio_processor.frames:
                # Combine all frames
                audio_data = np.concatenate(ctx.audio_processor.frames)

                # Convert to SR AudioData (48kHz → downsample to 16kHz if needed)
                audio = sr.AudioData(audio_data.tobytes(), sample_rate=48000, sample_width=2)

                try:
                    text = recognizer.recognize_google(audio, language="en-US")
                    st.success(f"🗣️ You said: {text}")
                except sr.UnknownValueError:
                    st.warning("⚠️ Could not understand the audio.")
                except sr.RequestError as e:
                    st.error(f"API error: {e}")

                # Clear buffer
                ctx.audio_processor.frames = []
            else:
                st.warning("⚠️ No audio captured yet.")
    return text

user_input = speech_to_text()
if user_input:
    st.write(f"Final Input: {user_input}")



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
            value="\n".join(
                [f"You: {m['content']}" if m['role']=='user' else f"AI: {m['content']}" for m in st.session_state.memory]
            ),
            height=300
        )

# --------------------------
# Voice Mode
# --------------------------
elif mode == "Voice":
    if st.button("Speak"):
        user_input = speech_to_text()
        if user_input:
            st.success(f"You said: {user_input}")
            response = run_conversation(user_input)
            st.text_area(
                "Conversation",
                value="\n".join(
                    [f"You: {m['content']}" if m['role']=='user' else f"AI: {m['content']}" for m in st.session_state.memory]
                ),
                height=300
            )
        else:
            st.error("⚠️ Could not recognize speech. Please try again.")

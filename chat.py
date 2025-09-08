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
# Voice Input Function
# ==========================
# (your imports + GroqLLM + run_conversation stay the same)
import os
import wget
import zipfile

# Make sure model directory exists only once
if not os.path.exists("model"):
    url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    zip_path = "vosk_model.zip"

    # Download
    wget.download(url, zip_path)

    # Extract
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")

    # Rename folder to "model"
    os.rename("vosk-model-small-en-us-0.15", "model")

    print("✅ Vosk model downloaded and extracted.")

# ==========================
# Voice Input Function
# ==========================


from vosk import Model, KaldiRecognizer
import json
import numpy as np

# Load Vosk model (make sure "model" folder exists in your project)
model = Model("model")

def speech_to_text_vosk(frames, sample_rate=16000):
    """
    Takes raw audio frames (numpy int16) and transcribes using Vosk.
    """
    recognizer = KaldiRecognizer(model, sample_rate)
    recognizer.SetWords(True)

    # Convert frames into raw bytes
    audio_bytes = frames.tobytes()

    if recognizer.AcceptWaveform(audio_bytes):
        result = json.loads(recognizer.Result())
        return result.get("text", "")
    else:
        partial = json.loads(recognizer.PartialResult())
        return partial.get("partial", "")
from vosk import Model, KaldiRecognizer
import json
import pyaudio

# Ensure Vosk model is downloaded at runtime (add this near the top of your code after imports)
import os, wget, zipfile

if not os.path.exists("model"):
    url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    zip_path = "vosk_model.zip"
    wget.download(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")

    os.rename("vosk-model-small-en-us-0.15", "model")

# Initialize Vosk model
model = Model("model")

def speech_to_text():
    rec = KaldiRecognizer(model, 16000)

    # Start PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=8192)
    stream.start_stream()

    st.info("🎙️ Listening... speak now!")

    text = ""
    while True:
        data = stream.read(4096, exception_on_overflow = False)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "")
            break  # stop after first full sentence

    stream.stop_stream()
    stream.close()
    p.terminate()

    if text.strip():
        return text
    else:
        st.warning("⚠️ Could not capture speech.")
        return None

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
    st.write("🎙️ Start speaking below...")

    user_input = speech_to_text()  # directly capture speech

    if user_input:  # If speech recognized
        st.success(f"You said: {user_input}")
        response = run_conversation(user_input)
        st.text_area(
            "Conversation",
            value="\n".join(
                [f"You: {m['content']}" if m['role']=='user' else f"AI: {m['content']}" for m in st.session_state.memory]
            ),
            height=300
        )

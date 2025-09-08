# streamlit_chat_text_or_voice.py
import streamlit as st
import speech_recognition as sr
from langchain_google_genai import ChatGoogleGenerativeAI
from litellm import completion
from langchain.schema import HumanMessage

# ==========================
# LLM Wrapper
# ==========================
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
class GroqLLM(ChatGoogleGenerativeAI):
    def __init__(self, **kwargs):
        super().__init__(model="gemini-1.5-flash", **kwargs)

    def _call(self, prompt: str, stop=None) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = completion(model="gemini-1.5-flash", messages=messages)
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
def speech_to_text():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        st.info("🎤 Listening... Speak now")
        audio = recognizer.listen(source, phrase_time_limit=5)
    try:
        return recognizer.recognize_google(audio)
    except:
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

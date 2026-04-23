import streamlit as st
from agent import run_agent
from rag import load_base_knowledge, add_user_pdf, get_retriever
from memory import save_message, load_history
import time
from gtts import gTTS
import os

st.set_page_config(page_title="AI AGENT", layout="wide")

# ---------------- STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "audio_file" not in st.session_state:
    st.session_state.audio_file = None

# ---------------- DB INIT ----------------
if "db_loaded" not in st.session_state:
    load_base_knowledge("data/Knowledge.pdf")
    st.session_state.retriever = get_retriever()
    st.session_state.db_loaded = True

user_id = "default_user"

# ---------------- TITLE ----------------
st.title("🌙 AI Assistant ✨")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("### 💖 Session")
    st.markdown(f"💬 Messages: {len(st.session_state.messages)}")

    uploaded = st.file_uploader("📂 Upload PDF", type="pdf")

    if uploaded:
        if uploaded.size > 5 * 1024 * 1024:
            st.error("File too large (max 5MB)")
        else:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded.read())

            add_user_pdf("temp.pdf")
            st.session_state.retriever = get_retriever()
            st.success("✨ PDF merged!")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- TTS ----------------
def speak(text):
    file_path = f"response_{int(time.time())}.mp3"
    tts = gTTS(text)
    tts.save(file_path)
    return file_path

# ---------------- DISPLAY ----------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"🧚‍♀️ {msg['content']}")
    else:
        st.markdown(f"🔮 {msg['content']}")

# ---------------- INPUT ----------------
user_input = st.chat_input("Ask something magical... ✨")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    save_message(user_id, "user", user_input)

    history = load_history(user_id)
    retriever = st.session_state.retriever

    with st.spinner("✨ Thinking..."):
        result = run_agent(user_input, retriever, history)

    display = ""
    output = st.empty()

    for ch in result:
        display += ch
        output.markdown(f"🔮 {display}")
        time.sleep(0.001)

    st.session_state.messages.append({"role": "assistant", "content": result})
    save_message(user_id, "assistant", result)

    st.session_state.audio_file = speak(result)

    st.rerun()

# ---------------- AUDIO ----------------
if st.session_state.audio_file:
    st.audio(st.session_state.audio_file)
import streamlit as st
from agent import run_agent
from rag import load_base_knowledge, add_user_pdf, get_retriever
from memory import save_message, load_history
import time
from gtts import gTTS
import speech_recognition as sr

st.set_page_config(page_title="AI AGENT", layout="wide")

# ---------------- INIT ----------------
if "retriever" not in st.session_state:
    load_base_knowledge()
    st.session_state.retriever = get_retriever()

if "messages" not in st.session_state:
    st.session_state.messages = []

user_id = "default_user"

# ---------------- CSS ----------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #1a0f2e, #0b0617);
}
h1 {
    text-align: center;
    color: #d6b3ff !important;
    text-shadow: 0 0 10px #b084ff, 0 0 25px rgba(176,132,255,0.6);
}
.user-msg {
    background: linear-gradient(135deg, #3b1e5c, #2a1442);
    padding: 12px;
    border-radius: 15px;
    margin: 10px 0;
    box-shadow: 0 0 12px #7a4cff;
}
.ai-msg {
    background: linear-gradient(135deg, #1e1335, #120a24);
    padding: 12px;
    border-radius: 15px;
    margin: 10px 0;
    box-shadow: 0 0 12px #b084ff;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #120a24, #0b0617);
    border-right: 1px solid #7a4cff;
}
section[data-testid="stSidebar"] * {
    color: #d6b3ff !important;
}
button {
    background: linear-gradient(135deg, #7a4cff, #b084ff);
    border-radius: 10px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🌙 AI Assistant ✨")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("### 💖 Session")
    st.markdown(f"💬 Messages: {len(st.session_state.messages)}")

    uploaded = st.file_uploader("📂 Upload PDF", type="pdf")

    if uploaded:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded.read())

        add_user_pdf("temp.pdf")
        st.session_state.retriever = get_retriever()
        st.success("✨ PDF merged!")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    # 🎤 safer voice input
    if st.button("🎤 Speak"):
        try:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                st.info("Listening...")
                audio = r.listen(source, timeout=5, phrase_time_limit=5)

            text = r.recognize_google(audio)
            st.session_state["voice_input"] = text

        except Exception as e:
            st.error(f"Voice failed: {e}")

# ---------------- VOICE FUNCTION ----------------
def speak(text):
    file_path = "response.mp3"
    tts = gTTS(text)
    tts.save(file_path)
    return file_path

# ---------------- DISPLAY ----------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'>🧚‍♀️ {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='ai-msg'>🔮 {msg['content']}</div>", unsafe_allow_html=True)

# ---------------- INPUT ----------------
user_input = st.chat_input("Ask something magical... ✨")

if "voice_input" in st.session_state:
    user_input = st.session_state.pop("voice_input")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    save_message(user_id, "user", user_input)

    # Load memory
    history = load_history(user_id)

    # Thinking
    thinking = st.empty()
    thinking.markdown("✨ Thinking...")
    time.sleep(0.3)
    thinking.empty()

    # ---------------- AGENT CALL (FIXED) ----------------
    result = run_agent(
        user_input,
        retriever=st.session_state.retriever,
        history=history
    )

    # Typing effect
    display = ""
    output = st.empty()

    for ch in result:
        display += ch
        output.markdown(
            f"<div class='ai-msg'>🔮 {display}</div>",
            unsafe_allow_html=True
        )
        time.sleep(0.002)

    # 🔊 Voice button
    if st.button("🔊 Play Voice"):
        audio_path = speak(result)
        st.audio(audio_path, format="audio/mp3")

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": result
    })
    save_message(user_id, "assistant", result)

    st.rerun()
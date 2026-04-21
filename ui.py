import streamlit as st
from agent import create_agent
from rag import load_default_doc, add_user_pdf, get_retriever
import time
from gtts import gTTS
import tempfile
import speech_recognition as sr

st.set_page_config(page_title="AI AGENT", layout="wide")

# ---------------- LOAD DEFAULT DOC ----------------
if "retriever" not in st.session_state:
    load_default_doc()
    st.session_state.retriever = get_retriever()

# ---------------- CSS ----------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #1a0f2e, #0b0617);
}

h1 {
    text-align: center;
    color: #d6b3ff !important;
    text-shadow: 0 0 15px #b084ff;
}

.user-msg {
    background: #3b1e5c;
    padding: 12px;
    border-radius: 15px;
    margin: 10px 0;
    box-shadow: 0 0 10px #7a4cff;
}

.ai-msg {
    background: #1e1335;
    padding: 12px;
    border-radius: 15px;
    margin: 10px 0;
    box-shadow: 0 0 10px #b084ff;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🌙 AI Assistant ✨")

# ---------------- AGENT ----------------
agent = create_agent()

# ---------------- SESSION ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("### 💖 Session")

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

    if st.button("🎤 Speak"):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening...")
            audio = r.listen(source)

        try:
            text = r.recognize_google(audio)
            st.session_state["voice_input"] = text
        except:
            st.error("Voice failed")

# ---------------- VOICE ----------------
def speak(text):
    tts = gTTS(text)
    file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(file.name)
    return file.name

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
    st.session_state.messages.append({"role": "user", "content": user_input})

    thinking = st.empty()
    thinking.markdown("✨ Thinking...")
    time.sleep(0.3)
    thinking.empty()

    response = agent.invoke({
        "query": user_input,
        "retriever": st.session_state.retriever
    })

    result = response["result"]

    output = st.empty()
    text = ""
    for ch in result:
        text += ch
        output.markdown(f"<div class='ai-msg'>🔮 {text}</div>", unsafe_allow_html=True)
        time.sleep(0.002)

    audio = speak(result)
    st.audio(audio)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result
    })

    st.rerun()
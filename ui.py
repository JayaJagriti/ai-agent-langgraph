import streamlit as st
from agent import create_agent
from rag import load_base_knowledge, add_user_pdf, get_retriever
from memory import save_message, load_history
import time
from gtts import gTTS
import tempfile
import speech_recognition as sr

st.set_page_config(page_title="AI AGENT", layout="wide")

# ---------------- 🌌 CSS (RESTORED GLOW UI) ----------------
st.markdown("""
<style>

/* 🌌 BACKGROUND */
.stApp {
    background: radial-gradient(circle at top, #1a0f2e, #0b0617) !important;
}

/* ✨ STARS */
.stApp::before {
    content: "";
    position: fixed;
    width: 100%;
    height: 100%;
    background-image: 
        radial-gradient(2px 2px at 20% 30%, #ffffff22, transparent),
        radial-gradient(1px 1px at 80% 70%, #ffffff33, transparent);
    animation: starsMove 60s linear infinite;
    z-index: -1;
}

@keyframes starsMove {
    from { transform: translateY(0px); }
    to { transform: translateY(-200px); }
}

/* 🌙 TITLE */
h1 {
    text-align: center;
    color: #d6b3ff !important;
    text-shadow:
        0 0 10px #b084ff,
        0 0 25px rgba(176,132,255,0.6);
}

/* 💬 USER MESSAGE */
.user-msg {
    background: linear-gradient(135deg, #3b1e5c, #2a1442);
    padding: 14px;
    border-radius: 20px;
    margin: 12px 0;
    box-shadow: 0 0 18px #7a4cff;
    color: white;
}

/* 🤖 AI MESSAGE */
.ai-msg {
    background: linear-gradient(135deg, #1e1335, #120a24);
    padding: 14px;
    border-radius: 20px;
    margin: 12px 0;
    box-shadow: 0 0 18px #b084ff;
    color: white;
}

/* 🔥 SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #120a24, #0b0617) !important;
    border-right: 1px solid #7a4cff !important;
}

/* SIDEBAR TEXT */
section[data-testid="stSidebar"] * {
    color: #d6b3ff !important;
}

/* 🔘 BUTTONS */
button[kind="secondary"] {
    background: linear-gradient(135deg, #7a4cff, #b084ff) !important;
    border-radius: 12px !important;
    color: white !important;
    border: none !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- INIT ----------------
if "retriever" not in st.session_state:
    load_base_knowledge()
    st.session_state.retriever = get_retriever()

if "messages" not in st.session_state:
    st.session_state.messages = []

agent = create_agent()
user_id = "default_user"

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

    if st.button("🔇 Stop Voice"):
        st.session_state["mute"] = True

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
    st.session_state["mute"] = False

    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    save_message(user_id, "user", user_input)

    # Load history
    history = load_history(user_id)

    # Thinking
    thinking = st.empty()
    thinking.markdown("✨ Thinking...")
    time.sleep(0.3)
    thinking.empty()

    # Agent call
    response = agent.invoke({
        "query": user_input,
        "retriever": st.session_state.retriever,
        "history": history
    })

    result = response["result"]

    # Typing effect
    display = ""
    output = st.empty()

    for ch in result:
        display += ch
        output.markdown(f"<div class='ai-msg'>🔮 {display}</div>", unsafe_allow_html=True)
        time.sleep(0.002)

    # Voice
    if not st.session_state.get("mute", False):
        audio = speak(result)
        st.audio(audio)

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": result
    })
    save_message(user_id, "assistant", result)

    st.rerun()
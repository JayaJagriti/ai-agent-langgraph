import streamlit as st
from agent import run_agent
from rag import load_base_knowledge, get_retriever

st.set_page_config(page_title="AI Agent", page_icon="🤖")

st.title("✨ Your Personal AI Assistant")

# Load knowledge once
if "retriever" not in st.session_state:
    db = load_base_knowledge("data/Knowledge.pdf")
    st.session_state.retriever = get_retriever(db)

# Store chat history
if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask something:")

if st.button("Run Agent"):
    if query:
        with st.spinner("Thinking..."):
            response = run_agent(
                query,
                retriever=st.session_state.retriever,
                history=st.session_state.history
            )

            # save memory
            st.session_state.history.append({"role": "user", "content": query})
            st.session_state.history.append({"role": "assistant", "content": response})

            st.write(response)
    else:
        st.warning("Enter something")
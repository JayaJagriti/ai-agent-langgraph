import streamlit as st
from agent import create_agent

# Page config
st.set_page_config(page_title="AI Agent", page_icon="🤖")

# Title
st.title("🤖 AI Agent (LangGraph + Groq)")

# Input box
query = st.text_input("Ask something:")

# Button (optional but cleaner UX)
if st.button("Run Agent"):
    if query:
        with st.spinner("Thinking..."):
            try:
                response = create_agent(query)
                st.success("Response:")
                st.write(response)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question.")
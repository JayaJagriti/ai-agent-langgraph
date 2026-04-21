import os
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_community.tools import DuckDuckGoSearchRun
from groq import Groq

from rag import debug_retrieval

load_dotenv()

# ---------------- LLM ----------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------- WEB TOOL ----------------
search = DuckDuckGoSearchRun()

# ---------------- STATE ----------------
class AgentState(TypedDict):
    query: str
    result: str
    retriever: object


# ---------------- ROUTER ----------------
def router(state: AgentState):
    query = state["query"].lower()

    # 🔥 smart routing
    if state.get("retriever"):
        if any(word in query for word in [
            "document", "pdf", "file", "explain", "summarize", "according"
        ]):
            return "rag"

    if any(word in query for word in [
        "latest", "news", "today", "current", "who is", "what is happening"
    ]):
        return "web"

    return "llm"


# ---------------- RAG NODE ----------------
def rag_node(state: AgentState):
    retriever = state.get("retriever")

    if not retriever:
        return {"result": "No document loaded."}

    docs = retriever.get_relevant_documents(state["query"])

    if not docs:
        return {"result": "No relevant info found in document."}

    # 🔥 keep top 3 only
    docs = docs[:3]

    # 🔍 debug
    debug_retrieval(docs)

    context = "\n\n".join([d.page_content for d in docs])

    if len(context.strip()) < 50:
        return {"result": "I could not find this in the document."}

    prompt = f"""
You are a strict document assistant.

Rules:
- ONLY use the context
- NO guessing
- NO outside knowledge
- If unclear → say "Not found in document"

CONTEXT:
{context}

QUESTION:
{state['query']}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"result": response.choices[0].message.content}


# ---------------- WEB NODE ----------------
def web_node(state: AgentState):
    try:
        result = search.run(state["query"])

        prompt = f"""
Use the web result below to answer clearly:

{result}

Question: {state['query']}
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        return {"result": response.choices[0].message.content}

    except:
        return {"result": "Web search failed."}


# ---------------- LLM NODE ----------------
def llm_node(state: AgentState):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful, smart AI assistant."
            },
            {
                "role": "user",
                "content": state["query"]
            }
        ]
    )

    return {"result": response.choices[0].message.content}


# ---------------- BUILD AGENT ----------------
def create_agent():
    builder = StateGraph(AgentState)

    builder.add_node("router", lambda state: state)
    builder.add_node("rag", rag_node)
    builder.add_node("web", web_node)
    builder.add_node("llm", llm_node)

    builder.set_entry_point("router")

    builder.add_conditional_edges(
        "router",
        router,
        {
            "rag": "rag",
            "web": "web",
            "llm": "llm"
        }
    )

    builder.add_edge("rag", END)
    builder.add_edge("web", END)
    builder.add_edge("llm", END)

    return builder.compile()
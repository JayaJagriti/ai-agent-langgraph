import os
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_community.tools import DuckDuckGoSearchRun
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
search = DuckDuckGoSearchRun()

# ---------------- STATE ----------------
class AgentState(TypedDict):
    query: str
    result: str
    retriever: object

# ---------------- SMART ROUTER ----------------
def smart_router(state: AgentState):
    query = state["query"]

    prompt = f"""
You are an AI router.

Decide best tool:
- "llm" → casual/general
- "rag" → document/pdf
- "web" → internet/recent info

Query: {query}

Return ONLY: llm / rag / web
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    decision = response.choices[0].message.content.strip().lower()

    if "rag" in decision and state.get("retriever"):
        return "rag"
    elif "web" in decision:
        return "web"
    return "llm"

# ---------------- LLM ----------------
def llm_node(state: AgentState):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": state["query"]}]
    )
    return {"result": response.choices[0].message.content}

# ---------------- RAG ----------------
def rag_node(state: AgentState):
    retriever = state.get("retriever")

    if not retriever:
        return {"result": "No document loaded."}

    docs = retriever.invoke(state["query"])

    if not docs:
        return {"result": "No relevant info found in document."}

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
Answer ONLY using this context:

{context}

Question: {state['query']}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"result": response.choices[0].message.content}

# ---------------- WEB ----------------
def web_node(state: AgentState):
    result = search.run(state["query"])

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": f"Summarize this clearly:\n{result}"
        }]
    )

    return {"result": response.choices[0].message.content}

# ---------------- BUILD ----------------
def create_agent():
    builder = StateGraph(AgentState)

    builder.add_node("router", lambda x: x)
    builder.add_node("llm", llm_node)
    builder.add_node("rag", rag_node)
    builder.add_node("web", web_node)

    builder.set_entry_point("router")

    builder.add_conditional_edges(
        "router",
        smart_router,
        {
            "llm": "llm",
            "rag": "rag",
            "web": "web"
        }
    )

    builder.add_edge("llm", END)
    builder.add_edge("rag", END)
    builder.add_edge("web", END)

    return builder.compile()
from typing import TypedDict
from langgraph.graph import StateGraph, END
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class AgentState(TypedDict):
    query: str
    result: str
    retriever: object
    history: list
    tool: str


# ---------------- ROUTER ----------------
def decide_tool(query, has_retriever, history):
    history_text = "\n".join([h["content"] for h in history[-4:]])

    prompt = f"""
You are an intelligent AI router.

Conversation:
{history_text}

User Query: {query}

Choose best tool:
- rag → if document related
- web → if latest/current info
- llm → normal/general

Return ONLY one word: rag / web / llm
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    out = res.choices[0].message.content.lower()

    if "rag" in out:
        return "rag"
    elif "web" in out:
        return "web"
    return "llm"


def router_node(state: AgentState):
    tool = decide_tool(
        state["query"],
        state.get("retriever") is not None,
        state.get("history", [])
    )
    return {"tool": tool}


# ---------------- LLM ----------------
def llm_node(state: AgentState):
    messages = state["history"] + [
        {"role": "user", "content": state["query"]}
    ]

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )

    return {"result": res.choices[0].message.content}


# ---------------- RAG ----------------
def rag_node(state: AgentState):
    retriever = state.get("retriever")

    if not retriever:
        return {"result": "No document loaded."}

    docs = retriever.invoke(state["query"])

    if not docs:
        docs = retriever.invoke("summary of document")

    context = "\n".join([d.page_content for d in docs])

    messages = state["history"] + [
        {
            "role": "user",
            "content": f"""
Answer using the context below:

{context}

Question: {state["query"]}
"""
        }
    ]

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )

    return {"result": res.choices[0].message.content}


# ---------------- GRAPH ----------------
def create_agent():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("llm", llm_node)
    graph.add_node("rag", rag_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        lambda s: s["tool"],
        {
            "llm": "llm",
            "rag": "rag",
            "web": "llm"  # fallback
        }
    )

    graph.add_edge("llm", END)
    graph.add_edge("rag", END)

    return graph.compile()
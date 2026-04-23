from typing import TypedDict
from langgraph.graph import StateGraph, END
import os
from groq import Groq
from dotenv import load_dotenv

# Load env
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------- STATE ----------------
class AgentState(TypedDict):
    query: str
    result: str
    retriever: object
    history: list
    tool: str


# ---------------- ROUTER ----------------
def decide_tool(query):
    prompt = f"""
You are a router.

Choose:
- rag → if answer is in internal company data
- llm → for general knowledge or casual queries

User Query: {query}

Return ONLY: rag or llm
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    out = res.choices[0].message.content.lower()

    if "rag" in out:
        return "rag"
    return "llm"


def router_node(state: AgentState):
    tool = decide_tool(state["query"])
    return {"tool": tool}


# ---------------- LLM ----------------
def llm_node(state: AgentState):
    history = state.get("history", [])[-4:]  # 🔥 trim history

    messages = history + [
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
        return llm_node(state)

    docs = retriever.invoke(state["query"])

    if not docs:
        return llm_node(state)

    context = "\n".join([d.page_content for d in docs])

    history = state.get("history", [])[-4:]  # 🔥 trim history

    messages = history + [
        {
            "role": "user",
            "content": f"""
Use the context if relevant, otherwise answer normally.

Context:
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
            "rag": "rag",
            "llm": "llm"
        }
    )

    graph.add_edge("rag", END)
    graph.add_edge("llm", END)

    return graph.compile()


# ---------------- RUN FUNCTION ----------------
def run_agent(query, retriever=None, history=None):
    agent = create_agent()

    result = agent.invoke({
        "query": query,
        "retriever": retriever,
        "history": history or []
    })

    return result["result"]
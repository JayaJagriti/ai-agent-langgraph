from typing import TypedDict
from langgraph.graph import StateGraph, END
import os
from groq import Groq
from dotenv import load_dotenv

# ---------------- SETUP ----------------
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
def decide_tool(query, has_retriever, history):
    history_text = "\n".join([h["content"] for h in history[-4:]])

    prompt = f"""
You are a smart AI router.

Conversation:
{history_text}

User Query: {query}

Rules:
- Use "rag" for company/internal data (meetings, employees, projects, policies)
- Use "llm" for general/world knowledge
- Prefer rag if unsure

Return ONLY:
rag or llm
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
    return {
        "tool": decide_tool(
            state["query"],
            state.get("retriever") is not None,
            state.get("history", [])
        )
    }

# ---------------- LLM ----------------
def llm_node(state: AgentState):
    messages = state.get("history", []) + [
        {"role": "user", "content": state["query"]}
    ]

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )

    return {"result": res.choices[0].message.content}

# ---------------- RAG (SAFE + CHAINED) ----------------
def rag_node(state: AgentState):
    retriever = state.get("retriever")

    # No retriever → fallback
    if not retriever:
        return llm_node(state)

    docs = retriever.invoke(state["query"])

    # ❌ No docs → fallback to LLM (general queries)
    if not docs or len(docs) == 0:
        return llm_node(state)

    context = "\n".join([d.page_content for d in docs])

    # 🧠 STEP 1: Check if context actually answers question
    check_prompt = f"""
You are a strict validator.

Context:
{context}

Question: {state["query"]}

Does the context clearly contain the answer?

Answer ONLY:
YES or NO
"""

    check = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": check_prompt}]
    )

    decision = check.choices[0].message.content.strip().lower()

    # ✅ CASE 1: GOOD RAG → answer strictly from context
    if "yes" in decision:
        messages = state.get("history", []) + [
            {
                "role": "user",
                "content": f"""
Answer ONLY from this internal knowledge.
Do NOT make up anything.

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

    # ❌ CASE 2: INTERNAL QUESTION BUT NOT FOUND → NO HALLUCINATION
    if any(word in state["query"].lower() for word in [
        "meeting", "employee", "project", "company", "policy"
    ]):
        return {
            "result": "I could not find this information in the internal database."
        }

    # 🔥 CASE 3: GENERAL QUESTION → TOOL CHAINING (RAG + LLM)
    messages = state.get("history", []) + [
        {
            "role": "user",
            "content": f"""
Internal data (may or may not help):
{context}

Use it if relevant. Otherwise answer using your knowledge.

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
        }
    )

    graph.add_edge("llm", END)
    graph.add_edge("rag", END)

    return graph.compile()

# ---------------- RUN ----------------
def run_agent(query, retriever=None, history=None):
    agent = create_agent()

    result = agent.invoke({
        "query": query,
        "retriever": retriever,
        "history": history or []
    })

    return result["result"]
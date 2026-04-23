from typing import TypedDict
from langgraph.graph import StateGraph, END
from groq import Groq
import os
from duckduckgo_search import DDGS

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------- STATE ----------------
class AgentState(TypedDict):
    query: str
    retriever: any
    history: list
    result: str


# ---------------- SMALL TALK ----------------
def is_small_talk(query: str):
    query = query.lower()
    return any(word in query for word in [
        "hi", "hello", "hey", "how are you", "what's up"
    ])


# ---------------- WEB SEARCH ----------------
def web_search(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))

        if results:
            return results[0]["body"]
    except:
        return "Web search failed."

    return "No relevant web results found."


# ---------------- LLM ----------------
def llm_node(state: AgentState):
    messages = state.get("history", [])
    messages.append({"role": "user", "content": state["query"]})

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )

    return {"result": res.choices[0].message.content}


# ---------------- RAG NODE ----------------
def rag_node(state: AgentState):
    query = state["query"]
    retriever = state.get("retriever")

    # 👉 small talk → LLM
    if is_small_talk(query):
        return llm_node(state)

    # 👉 RAG first
    if retriever:
        docs = retriever.invoke(query)

        if docs:
            context = "\n".join([d.page_content for d in docs])

            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{
                    "role": "user",
                    "content": f"""
You are a helpful assistant.

Use the context to answer the question.
If the answer can be inferred, answer clearly.

If not present, say: "NOT_FOUND"

Context:
{context}

Question:
{query}
"""
                }]
            )

            answer = res.choices[0].message.content

            # 👉 if not found → web search
            if "NOT_FOUND" not in answer:
                return {"result": answer}

    # 👉 WEB SEARCH FALLBACK
    web_result = web_search(query)

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": f"""
Answer using this web result:

{web_result}

Question:
{query}
"""
        }]
    )

    return {"result": res.choices[0].message.content}


# ---------------- GRAPH ----------------
def create_agent():
    graph = StateGraph(AgentState)
    graph.add_node("rag", rag_node)
    graph.set_entry_point("rag")
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
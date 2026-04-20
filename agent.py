from google import genai
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_community.tools import DuckDuckGoSearchRun
from rag import create_retriever

# 🔑 API KEY
client = genai.Client(api_key="AIzaSyBiYWdDRh8NLljbPJbhnQC4iQIw9VWypcQ")

# State
class AgentState(TypedDict):
    query: str
    result: str

# Tools
retriever = create_retriever()
search = DuckDuckGoSearchRun()

# Router
def router(state: AgentState):
    query = state["query"].lower()
    if "latest" in query or "news" in query:
        return "web"
    return "rag"

# RAG node
def rag_node(state: AgentState):
    docs = retriever.invoke(state["query"])
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer using the context below:

{context}

Question: {state['query']}
"""

    response = client.models.generate_content(
        model="models/gemini-flash-latest",
        contents=prompt,
    )

    return {"result": response.text}

# Web node
def web_node(state: AgentState):
    result = search.run(state["query"])
    return {"result": result}

# ✅ IMPORTANT — THIS FUNCTION MUST EXIST
def create_agent():
    builder = StateGraph(AgentState)

    builder.add_node("router", lambda state: state)
    builder.add_node("rag", rag_node)
    builder.add_node("web", web_node)

    builder.set_entry_point("router")

    builder.add_conditional_edges(
        "router",
        router,
        {
            "rag": "rag",
            "web": "web",
        }
    )

    builder.add_edge("rag", END)
    builder.add_edge("web", END)

    return builder.compile()
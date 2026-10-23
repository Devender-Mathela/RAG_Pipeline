from langgraph.graph import StateGraph, START, END

from agents.state import AgentState
from agents.nodes import (
    router,
    retrieve,
    grade_documents,
    rewrite_query,
    web_search,
    generate,
    verify_evidence
)

def decide_to_generate(state):
    """
    Conditional routing after document grading:
    1. If relevant documents found -> proceed to generate.
    2. If irrelevant and retries < 1 -> rewrite query & search again.
    3. If irrelevant and retries exhausted -> escalate to web search fallback.
    """
    if state.get("documents_relevant", False):
        return "generate"
    elif state.get("retry_count", 0) < 1:
        return "rewrite_query"
    else:
        return "web_search"


builder = StateGraph(AgentState)

# Add all nodes
builder.add_node("router", router)
builder.add_node("retrieve", retrieve)
builder.add_node("grade_documents", grade_documents)
builder.add_node("rewrite_query", rewrite_query)
builder.add_node("web_search", web_search)
builder.add_node("generate", generate)
builder.add_node("verify_evidence", verify_evidence)

# Flow: START -> router
builder.add_edge(START, "router")

# Router conditional branch (rag vs web)
builder.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {
        "rag": "retrieve",
        "web": "web_search"
    }
)

# Retrieval -> Grade Documents
builder.add_edge("retrieve", "grade_documents")

# Grade Documents -> Generate OR Rewrite Query OR Web Fallback
builder.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "rewrite_query": "rewrite_query",
        "web_search": "web_search"
    }
)

# Loop: Rewrite Query -> Retrieve
builder.add_edge("rewrite_query", "retrieve")

# Generation & Verification
builder.add_edge("web_search", "generate")
builder.add_edge("generate", "verify_evidence")
builder.add_edge("verify_evidence", END)

graph = builder.compile()



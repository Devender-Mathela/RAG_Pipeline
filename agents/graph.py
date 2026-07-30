from langgraph.graph import StateGraph, START, END

from agents.state import AgentState
from agents.nodes import (
    router,
    retrieve,
    web_search,
    generate
)

builder=StateGraph(AgentState)

builder.add_node("router", router)
builder.add_node("retrieve",retrieve)
builder.add_node("web_search", web_search)
builder.add_node("generate",generate)

builder.add_edge(START,"router")
builder.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {
        "rag": "retrieve",
        "web": "web_search"
    }
)
builder.add_edge("retrieve", "generate")
builder.add_edge("web_search","generate")
builder.add_edge("generate",END)

graph=builder.compile()


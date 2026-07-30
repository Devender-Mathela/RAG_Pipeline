from retrieval.hybrid_search import hybrid_search
from rag.prompt import RAG_PROMPT
from llm.model import load_llm
from agents.tools import web_search_tool
from agents.prompts import ROUTER_PROMPT

llm=load_llm()

def web_search(state):

    question = state["question"]

    results = web_search_tool.invoke(question)
    
    # Handle case where TavilySearchResults returns a JSON string
    if isinstance(results, str):
        import json
        try:
            results = json.loads(results)
        except Exception:
            # If it's a string but not JSON, just wrap it in a dict
            results = [{"content": results, "title": "Web Search Result", "url": ""}]

    documents = []

    context = ""

    for result in results:

        text = result.get("content", "")

        title = result.get("title", "Unknown")

        context += (
            f"Title: {title}\n"
            f"Content: {text}\n\n"
        )

        documents.append(
            {
                "text": text,
                "metadata": {
                    "source": result.get(
                        "url",
                        "Web Source"
                    ),
                    "title": result.get(
                        "title",
                        "Unknown"
                    )
                }
            }
        )

    return {
        "context": context,
        "documents": documents
    }

def retrieve(state):

    question=state['question']
    documents=hybrid_search(state["question"])
    context="\n\n".join(doc['text'] for doc in documents)
    return{
        "documents":documents,
        "context":context
    }

def router(state):

    question = state["question"]

    prompt = ROUTER_PROMPT.invoke(
        {
            "question": question
        }
    )

    response = llm.invoke(prompt)

    route = response.content.strip().lower()

    if route not in ["rag", "web"]:
        route = "rag"

    return {
        "route": route
    }


def generate(state):

    prompt=RAG_PROMPT.invoke({
        "context":state["context"],
        "question":state["question"]
    })

    response=llm.invoke(prompt)

    return{
        "answer":response.content
    }
from retrieval.hybrid_search import hybrid_search
from rag.prompt import RAG_PROMPT
from llm.model import load_llm
from agents.tools import web_search_tool
from agents.prompts import ROUTER_PROMPT, GRADER_PROMPT, REWRITE_PROMPT, VERIFY_EVIDENCE_PROMPT

llm = load_llm()


def grade_documents(state):
    """
    Evaluates whether retrieved documents contain relevant information.
    """
    documents = state.get("documents", [])
    context = state.get("context", "").strip()

    if not documents or not context:
        return {"documents_relevant": False}

    # Pass the question and context into the grader prompt
    prompt = GRADER_PROMPT.invoke({
        "question": state["question"],
        "context": context[:1500]  # truncate to keep tokens low & cost minimal
    })

    response = llm.invoke(prompt)
    verdict = response.content.strip().lower()
    is_relevant = "yes" in verdict

    return {
        "documents_relevant": is_relevant
    }


def rewrite_query(state):
    """
    Transforms a failed search query into a better formulated search query.
    """
    current_retries = state.get("retry_count", 0) + 1
    original_q = state.get("original_question", state["question"])

    prompt = REWRITE_PROMPT.invoke({
        "question": state["question"]
    })

    response = llm.invoke(prompt)
    rewritten_q = response.content.strip().strip('"')

    return {
        "question": rewritten_q,
        "original_question": original_q,
        "retry_count": current_retries
    }


def generate(state):
    prompt = RAG_PROMPT.invoke({
        "context": state["context"],
        "question": state.get("original_question", state["question"])
    })

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


def verify_evidence(state):
    """
    Evidence-based answer verification guardrail:
    Checks if the draft answer is faithfully grounded in the retrieved context.
    """
    context = state.get("context", "").strip()
    answer = state.get("answer", "").strip()

    if not context:
        disclaimer = "\n\n> ⚠️ **Evidence Notice:** No document context was found to support this answer."
        return {
            "answer": answer + disclaimer,
            "evidence_status": "INSUFFICIENT_EVIDENCE",
            "evidence_reason": "No relevant context available to verify claims."
        }

    prompt = VERIFY_EVIDENCE_PROMPT.invoke({
        "context": context[:2000],
        "answer": answer
    })

    response = llm.invoke(prompt)
    content = response.content.strip()

    is_verified = "STATUS: VERIFIED" in content or content.startswith("VERIFIED")

    reason = ""
    for line in content.splitlines():
        if "REASON:" in line:
            reason = line.split("REASON:", 1)[1].strip()
            break
    if not reason:
        reason = content

    if is_verified:
        return {
            "evidence_status": "VERIFIED",
            "evidence_reason": reason
        }
    else:
        disclaimer = "\n\n> ⚠️ **Evidence Notice:** Certain claims could not be verified directly against the ingested documents."
        return {
            "answer": answer + disclaimer,
            "evidence_status": "UNVERIFIED",
            "evidence_reason": reason
        }


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

from retrieval.adaptive_retriever import adaptive_search

def retrieve(state):
    question = state["question"]
    search_result = adaptive_search(question, k=5, enable_rerank=True)
    documents = search_result["documents"]
    context = "\n\n".join(doc["text"] for doc in documents)
    return {
        "documents": documents,
        "context": context,
        "search_strategy": search_result["strategy"]
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
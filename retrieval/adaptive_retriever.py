import re
from typing import Dict, Any, List
from retrieval.vector_search import vector_search
from retrieval.keyword_search import keyword_search
from retrieval.hybrid_search import hybrid_search
from retrieval.reranker import rerank_documents


def classify_query_intent(query: str) -> Dict[str, str]:
    """
    Analyzes query characteristics to select the most optimal retrieval strategy.
    
    Zero-cost heuristic classification:
    - Keyword Search: Exact quotes, technical IDs/error codes, version numbers, or acronyms.
    - Vector Search: Conceptual, broad explanatory, or semantic relationship questions.
    - Hybrid Search: Default balanced approach combining both exact terms and conceptual context.
    """
    query_clean = query.strip()

    # 1. Check for exact quotes: "..." or '...'
    has_quotes = bool(re.search(r'["\'].+?["\']', query_clean))

    # 2. Check for technical codes, uppercase acronyms, or version numbers (e.g., ERR_404, v2.1, API_KEY)
    has_technical_code = bool(re.search(r'\b[A-Z0-9_]{3,}\b|\bv?\d+\.\d+\b', query_clean))

    # 3. Check for conceptual or explanatory questions
    conceptual_markers = [
        "explain", "what is the concept", "how does", "why does",
        "difference between", "overview", "relationship between", "describe"
    ]
    is_conceptual = any(query_clean.lower().startswith(marker) for marker in conceptual_markers)

    # Strategy Selection Logic
    if has_quotes or has_technical_code:
        return {
            "strategy": "keyword",
            "reason": "Query contains exact phrases, codes, or technical identifiers best matched by BM25."
        }
    elif is_conceptual and not has_technical_code:
        return {
            "strategy": "vector",
            "reason": "Query is conceptual or exploratory, best matched by dense vector embeddings."
        }
    else:
        return {
            "strategy": "hybrid",
            "reason": "Query benefits from both semantic context and keyword presence via Reciprocal Rank Fusion."
        }


def adaptive_search(query: str, k: int = 5, candidate_pool: int = 10, enable_rerank: bool = True) -> Dict[str, Any]:
    """
    Dynamically routes query to vector, keyword, or hybrid search, then reranks candidates.

    Args:
        query: User question
        k: Number of final documents to return
        candidate_pool: Number of candidate documents retrieved before reranking
        enable_rerank: Whether to apply Cross-Encoder reranking to candidates

    Returns:
        Dict containing the strategy used, reason, and list of top documents.
    """
    # Step 1: Detect best strategy
    intent = classify_query_intent(query)
    strategy = intent["strategy"]
    reason = intent["reason"]

    # Step 2: Retrieve candidates based on strategy
    if strategy == "keyword":
        candidates = keyword_search(query, k=candidate_pool)
    elif strategy == "vector":
        candidates = vector_search(query, k=candidate_pool)
    else: # "hybrid"
        candidates = hybrid_search(query, k=candidate_pool, candidate_pool=candidate_pool)

    # Step 3: Apply Cross-Encoder Reranker if requested
    if enable_rerank and candidates:
        final_docs = rerank_documents(query=query, documents=candidates, top_k=k)
    else:
        final_docs = candidates[:k]

    return {
        "query": query,
        "strategy": strategy,
        "reason": reason,
        "documents": final_docs
    }


if __name__ == "__main__":
    queries = [
        'Find code "ERR_502_BAD_GATEWAY"',
        "Explain the intuition behind self-attention mechanisms",
        "How is MongoDB Atlas used in Hybrid Search RAG?"
    ]

    for q in queries:
        decision = classify_query_intent(q)
        print(f"\nQuery: {q}")
        print(f"-> Strategy: {decision['strategy']} ({decision['reason']})")

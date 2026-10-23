from retrieval.vector_search import vector_search
from retrieval.keyword_search import keyword_search

def reciprocal_rank_fusion(vector_results: list, keyword_results: list, k: int = 5, c: int = 60) -> list:
    """
    Combines ranked results using Reciprocal Rank Fusion (RRF).
    
    Formula:
        RRF_Score(doc) = sum(1 / (c + rank))
        where 'c' is a constant (typically 60) that balances high and low ranking documents.
    """
    scores = {}
    doc_lookup = {}

    # Score vector results
    for rank, doc in enumerate(vector_results, start=1):
        text = doc.get("text", "")
        if not text:
            continue
        scores[text] = scores.get(text, 0.0) + (1.0 / (c + rank))
        if text not in doc_lookup:
            doc_lookup[text] = doc

    # Score keyword results
    for rank, doc in enumerate(keyword_results, start=1):
        text = doc.get("text", "")
        if not text:
            continue
        scores[text] = scores.get(text, 0.0) + (1.0 / (c + rank))
        if text not in doc_lookup:
            doc_lookup[text] = doc

    # Sort documents by RRF score in descending order
    sorted_texts = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)

    fused_results = []
    for text in sorted_texts[:k]:
        doc = dict(doc_lookup[text])
        doc["rrf_score"] = round(scores[text], 5)
        fused_results.append(doc)

    return fused_results


def hybrid_search(query: str, k: int = 5, candidate_pool: int = 10) -> list:
    """
    Combine Vector Search + Keyword Search results using Reciprocal Rank Fusion (RRF).

    Args:
        query: User query
        k: Number of final results
        candidate_pool: Number of candidate chunks retrieved from each source before fusion

    Returns:
        Combined and reranked relevant chunks
    """
    # 1. Get semantic candidates
    vector_results = vector_search(query, k=candidate_pool)

    # 2. Get keyword candidates
    keyword_results = keyword_search(query, k=candidate_pool)

    # 3. Fuse via Reciprocal Rank Fusion
    return reciprocal_rank_fusion(vector_results, keyword_results, k=k)


if __name__ == "__main__":
    query = "What is self attention?"
    results = hybrid_search(query, k=5)

    for index, result in enumerate(results, start=1):
        print(f"\nResult {index} | RRF Score: {result.get('rrf_score')}")
        print("-" * 40)
        print(result.get("text")[:200] + "...")


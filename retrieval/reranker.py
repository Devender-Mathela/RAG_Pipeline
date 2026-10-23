import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# Global variable to cache the CrossEncoder model in memory
_reranker_model = None

def get_reranker_model():
    """
    Lazy loader for the CrossEncoder model.
    Loads 'cross-encoder/ms-marco-MiniLM-L-6-v2' (lightweight, ~80MB, fast on CPU, 0 API cost).
    """
    global _reranker_model
    if _reranker_model is None:
        try:
            from sentence_transformers import CrossEncoder
            # Load cross-encoder model
            _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
            logger.info("CrossEncoder model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load CrossEncoder model ({e}). Will fallback to RRF score.")
            _reranker_model = False
    return _reranker_model


def rerank_documents(query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Rerank candidate documents against the user query using a Cross-Encoder.

    Args:
        query: User question
        documents: List of retrieved document dictionaries containing 'text'
        top_k: Number of top documents to keep after reranking

    Returns:
        Sorted list of top_k documents with an added 'rerank_score'
    """
    if not documents:
        return []

    model = get_reranker_model()

    # Fallback if cross-encoder model cannot be loaded
    if not model:
        for doc in documents:
            doc.setdefault("rerank_score", doc.get("rrf_score", doc.get("score", 0.0)))
        return documents[:top_k]

    # Prepare [query, text] pairs for cross-encoding
    pairs = [[query, doc.get("text", "")] for doc in documents]

    try:
        # CrossEncoder predicts direct relevance logits
        scores = model.predict(pairs)

        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(round(score, 4))

        # Sort descending by rerank_score
        reranked_docs = sorted(
            documents,
            key=lambda d: d.get("rerank_score", 0.0),
            reverse=True
        )

        return reranked_docs[:top_k]

    except Exception as e:
        logger.error(f"Error during reranking execution: {e}")
        return documents[:top_k]


if __name__ == "__main__":
    test_query = "What is transformer self attention?"
    sample_docs = [
        {"text": "Python is a programming language used for general purpose software.", "source": "doc1"},
        {"text": "Transformers rely on self-attention mechanisms to compute representations of input sequences.", "source": "doc2"},
        {"text": "Cooking pasta requires boiling water and adding salt.", "source": "doc3"}
    ]

    print("Reranking test...")
    results = rerank_documents(test_query, sample_docs, top_k=2)
    for r in results:
        print(f"Score: {r.get('rerank_score')} | Text: {r.get('text')}")

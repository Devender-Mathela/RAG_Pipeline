from retrieval.vector_search import vector_search
from retrieval.keyword_search import keyword_search

def hybrid_search(query:str, k:int =5
                  ):
    """
    Combine Vector Search + Keyword Search results.

    Args:
        query: User query
        k: Number of final results

    Returns:
        Combined relevant chunks
    """

    #Get semantic results
    vector_results=vector_search(query, k=k)

    #Get keyword results
    keyword_results=keyword_search(query,k=k)

    combined_results=[]

    seen_texts=set()

    # Add vector search results first
    for result in vector_results:

        text=result["text"]

        if text not in seen_texts:
            combined_results.append(result)
            seen_texts.add(text)

    # Add keyword search results
    for result in keyword_results:

        text=result["text"]

        if text not in seen_texts:
            combined_results.append(result)
            seen_texts.add(text)

    return combined_results[:k]

if __name__=="__main__":
    query="What is self attention?"

    results=hybrid_search(query,k=5)

    for index, result in enumerate(results, start=1):
        print("\nResult:", index)
        print("-"*40)
        print(result)

from database.mongodb import get_collection

def keyword_search(query:str, k:int=5):
    """
    Perform keyword search using MONGODB Atlas Search (BM25).

    Args:
        query: User question
        k: Number of results

    Returns:
        List of matched documents
    """

    collection=get_collection()

    pipeline = [
        {
            "$search": {
                "index": "keyword_index",
                "text": {
                    "query": query,
                    "path": "text"
                }
            }
        },
        {
            "$limit": k
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "metadata": 1,
                "score": {
                    "$meta": "searchScore"
                }
            }
        }
    ]

    results= list(collection.aggregate(pipeline))

    return results

if __name__=="__main__":
    query="What is self_attention ?"

    results= keyword_search(query)

    for i, result in enumerate(results, start=1):

        print(f"\nResult {i}")
        print("-"*40)
        print("Score:",result['score'])
        print(result['text'])

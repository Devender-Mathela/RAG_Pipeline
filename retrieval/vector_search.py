from langchain_huggingface import HuggingFaceEmbeddings
from database.mongodb import get_collection

#load embedding model once
embedding_model=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def vector_search(query:str, k:int=5):
    """
    Performs semantic search using MongoDB Atlas Vector Search.

    Args:
        query:User question
        k: Number of results

    Returns:
        List of relevant chunks
    """

    print("Step 1")
    collection=get_collection()

    #convert query into embeddings
    print("Step 2")
    query_embedding=embedding_model.embed_query(query)

    print("Step 3")
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",          # Atlas Vector Search index name
                "path": "embedding",              # Field containing embeddings
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": k
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "metadata": 1,
                "score": {
                    "$meta": "vectorSearchScore"
                }
            }
        }
    ]

    print("Step 4")
    results=list(collection.aggregate(pipeline))

    print("Step 5")
    return results

if __name__=="__main__":

    query="What is self attention?"

    results=vector_search(query)

    for i, result in enumerate(results,start=1):
        print(f"\nResult {i}")
        print("-"*40)
        print(f"Score:",{result["score"]})
        print(result["text"])
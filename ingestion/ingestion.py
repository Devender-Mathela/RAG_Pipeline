from database.mongodb import get_collection
from ingestion.loader import load_document
from ingestion.chunker import chunk_documents
from ingestion.embedding import create_embeddings
from ingestion.preprocessing import preprocess_documents


def store_in_mongodb(documents):

    collection=get_collection()

    if documents:
        # Clear all previous records before inserting new ones
        collection.delete_many({})
        print("Cleared previous records from the database.")
        
        results=collection.insert_many(
            documents
        )
        print(results.inserted_ids)
        print(collection.count_documents({}))
    


#complete Ingestion Pipeline

def ingest(file_path):



    print("Loading Document...")
    documents=load_document(file_path)

    print("Preprocessing Document...")
    documents=preprocess_documents(documents)

    print("Create semantic chunks")
    chunks=chunk_documents(documents)

    print(f"Created {len(chunks)} chunnks")

    print("Creating Embeddings...")
    embedded_documents=create_embeddings(chunks)

    print("Storing in MongoDB..")
    store_in_mongodb(embedded_documents)

    print("Ingestion completed successfully")


if __name__=="__main__":
    ingest("data/documents/self-attention.pdf")
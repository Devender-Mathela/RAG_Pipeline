from langchain_huggingface import HuggingFaceEmbeddings


# Initialize embedding model once
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def create_embeddings(documents):
    """
    Generate embeddings for document chunks.

    Args:
        documents:
            List of LangChain Document objects

    Returns:
        List of dictionaries containing:
        text, metadata, embedding
    """

    embedded_documents = []


    texts = [
        doc.page_content
        for doc in documents
    ]


    # Generate embeddings in batch
    vectors = embedding_model.embed_documents(
        texts
    )


    for doc, vector in zip(
        documents,
        vectors
    ):

        embedded_documents.append(
            {
         

                "text": doc.page_content,

                "embedding": vector,

                "metadata": doc.metadata
            }
        )


    return embedded_documents
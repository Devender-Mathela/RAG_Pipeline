from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion.embedding import embedding_model


# Semantic chunker
semantic_chunker = SemanticChunker(
    embedding_model,
    breakpoint_threshold_type="percentile"
)


# Fallback chunker
recursive_chunker = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


def chunk_documents(documents):

    """
    Chunk documents according to file type.
    """

    if not documents:
        return []


    file_type = documents[0].metadata.get(
        "file_type",
        ""
    )


    # -------------------------
    # CSV
    # -------------------------

    if file_type == ".csv":

        return csv_chunker(documents)


    # -------------------------
    # PDF, TXT, DOCX
    # -------------------------

    elif file_type in [
        ".pdf",
        ".txt",
        ".docx"
    ]:

        return semantic_chunker.split_documents(
            documents
        )


    # -------------------------
    # Default
    # -------------------------

    else:

        return recursive_chunker.split_documents(
            documents
        )


def csv_chunker(documents):

    """
    CSV works better with row-based chunks.
    """

    chunks = []

    for doc in documents:

        chunks.append(doc)


    return chunks
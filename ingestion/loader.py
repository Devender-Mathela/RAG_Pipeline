import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
)


def load_document(file_path: str):

    """
    Load different document types and return LangChain Documents.

    Supported:
    - PDF
    - CSV
    - TXT
    - DOCX
    - HTML
    """

    extension = os.path.splitext(file_path)[1].lower()


    if extension == ".pdf":

        loader = PyPDFLoader(
            file_path
        )


    elif extension == ".csv":

        loader = CSVLoader(
            file_path=file_path,
            encoding="utf-8"
        )


    elif extension == ".txt":

        loader = TextLoader(
            file_path,
            encoding="utf-8"
        )


    elif extension == ".docx":

        loader = Docx2txtLoader(
            file_path
        )


    elif extension in [".html", ".htm"]:

        loader = UnstructuredHTMLLoader(
            file_path
        )


    else:

        raise ValueError(
            f"Unsupported file type: {extension}"
        )


    documents = loader.load()


    # Add common metadata
    for doc in documents:

        doc.metadata["file_type"] = extension
        doc.metadata["source"] = file_path


    return documents
import re


def clean_text(text: str) -> str:
    """
    Clean raw document text.
    """

    if not text:
        return ""


    # Remove extra spaces and newlines
    text = re.sub(
        r"\s+",
        " ",
        text
    )


    # Remove multiple dots
    text = re.sub(
        r"\.{2,}",
        ".",
        text
    )


    # Strip spaces
    text = text.strip()


    return text



def preprocess_documents(documents):
    """
    Apply preprocessing to LangChain Documents.
    """

    processed_documents = []


    for doc in documents:

        cleaned_text = clean_text(
            doc.page_content
        )


        # Update document content
        doc.page_content = cleaned_text


        processed_documents.append(
            doc
        )


    return processed_documents
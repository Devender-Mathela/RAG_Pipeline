from langchain_core.prompts import ChatPromptTemplate


RAG_PROMPT = ChatPromptTemplate.from_template(
    """
    You are an AI assistant that answers questions using ONLY the provided context.

    Instructions:
    1. Read the context carefully before answering.
    2. Answer only from the provided context.
    3. Do not use outside knowledge.
    4. If the answer is not available in the context, reply:
    "I couldn't find the answer in the provided documents."
    5. Keep your answer clear and concise.
    6. If appropriate, use bullet points.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)
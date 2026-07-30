from langchain_core.prompts import ChatPromptTemplate

ROUTER_PROMPT = ChatPromptTemplate.from_template(
"""
You are a routing assistant.

Your job is to decide where the user's question should be answered from.

Rules:

1. DEFAULT to "rag" for almost all questions. If the user is asking about a concept, asking for a summary, or asking a question that could reasonably be answered by a provided document, choose "rag".

Examples of "rag" questions:
- What is the definition of [concept]?
- Summarize the main points of the uploaded document.
- What does chapter 3 discuss?
- How does the system described in the document work?

2. Choose "web" ONLY if the question explicitly requires real-time information, current events, or is obviously outside the scope of any static document.

Examples of "web" questions:
- Latest news about [topic]
- Today's weather in [city]
- Who won the recent sports match?
- Current stock price of [company]

Return ONLY one word:

rag

or

web

Question:
{question}
"""
)
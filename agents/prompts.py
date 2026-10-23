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


GRADER_PROMPT = ChatPromptTemplate.from_template(
"""You are an evaluator assessing relevance of retrieved document context to a user question.

Context:
{context}

Question:
{question}

Does the retrieved context contain information relevant to answering the user question?
Answer strictly with one word: "yes" or "no".
"""
)


REWRITE_PROMPT = ChatPromptTemplate.from_template(
"""You are a search query optimizer for a RAG retrieval system.
The original query failed to retrieve relevant documents. Rewrite the query to be clearer, more specific, and optimized for search retrieval.

Original Query:
{question}

Provide ONLY the rewritten query text with no introduction, explanation, or quotes.
"""
)


VERIFY_EVIDENCE_PROMPT = ChatPromptTemplate.from_template(
"""You are an evidence verification guardrail. Your job is to check whether the draft answer is faithfully grounded in and supported by the provided context.

Context:
{context}

Draft Answer:
{answer}

Instructions:
1. If the claims in the answer are supported by the context, reply with:
STATUS: VERIFIED
REASON: [One concise sentence explaining support]

2. If the answer contains claims NOT supported by the context or claims the context does not contain the answer, reply with:
STATUS: UNVERIFIED
REASON: [One concise sentence explaining what is missing or unsupported]
"""
)
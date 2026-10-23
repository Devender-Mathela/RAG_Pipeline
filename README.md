# 🛡️ Evidence-Driven Knowledge Intelligence Agent
### Adaptive Hybrid RAG • Corrective Retrieval (CRAG) • Evidence Verification Guardrail

A production-grade Agentic Retrieval-Augmented Generation (RAG) system built with **LangGraph**, **MongoDB Atlas**, and **Groq (Qwen-2.5)** that knows when search results are insufficient, self-heals its retrieval strategy, and verifies whether generated answers are strictly grounded in evidence before responding.

---

## 🌟 Key Differentiating Features

### 1. 🎯 Adaptive Retrieval (Zero-Cost Intent Routing + RRF + Cross-Encoder)
* **Intent-Aware Routing:** Automatically detects query patterns (exact quotes, error codes, technical IDs, or conceptual questions) and routes between BM25 keyword search, dense vector embeddings, or hybrid search.
* **Reciprocal Rank Fusion (RRF):** Merges dense and sparse candidates with standard mathematical ranking ($k=60$).
* **Local Cross-Encoder Reranker:** Runs `cross-encoder/ms-marco-MiniLM-L-6-v2` locally on CPU with **$0 API cost** to eliminate context noise.

### 2. 🛡️ Corrective Retrieval (CRAG) & Evidence Verification Guardrail
* **Document Relevance Grader:** Filters out low-quality retrieval before generation.
* **Self-Healing Query Rewriter:** If retrieved passages are insufficient, reformulates the search query and retries (strictly capped at 1 retry for cost/loop safety), automatically falling back to Tavily Web Search if necessary.
* **Evidence Verification Guardrail:** Analyzes draft answers against the context. If claims are unsupported or hallucinated, flags them as unverified and attaches cautionary evidence notices rather than fabricating facts.

### 3. 🔍 Deep Observability & Evidence Inspector (Streamlit UI)
* Real-time inspection of routing decisions, document pass/fail grading, and candidate chunk reranking scores.
* Transparent evidence grounding breakdown and query auto-correction telemetry.

---

## 🏗️ Architecture

```text
                   User Query
                       │
                       ▼
                 Streamlit UI
                       │
                       ▼
              LangGraph Agent Router
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
    Adaptive Retrieval      Tavily Web
   (Dense + BM25 + RRF)          │
             │                   │
             ▼                   │
    Cross-Encoder Rerank         │
             │                   │
             ▼                   │
    Document Grader Node         │
     (Relevant? Yes/No)          │
       ├── Low Relevance ──► [Rewrite Query] ──► Retry
       └── Exceeded ────────► Fallback to Web
             │
             ▼
      LLM Generation
             │
             ▼
  Evidence Verifier Guardrail
    (Checks Grounded Claims)
             │
             ▼
Verified Answer + Citation Sources
```

---

## 📁 Project Structure

```text
Hybrid_Search_RAG/
├── agents/
│   ├── graph.py               # LangGraph StateGraph with CRAG loops & verification
│   ├── nodes.py               # Grader, Rewriter, Generator, Verifier
│   ├── prompts.py             # Concise, low-token prompts
│   ├── state.py               # TypedDict state holding verification & strategy
│   └── tools.py               # Tavily search tool
│
├── database/
│   └── mongodb.py             # MongoDB Atlas connection
│
├── ingestion/
│   ├── loader.py              # Multi-format document loader
│   ├── preprocessing.py       # Text cleaning & normalization
│   ├── chunker.py             # Semantic Chunking + CSV Chunker
│   ├── embedding.py           # HuggingFace all-MiniLM-L6-v2 embeddings
│   └── ingestion.py           # Atlas ingestion pipeline
│
├── llm/
│   └── model.py               # ChatGroq LLM initialization
│
├── retrieval/
│   ├── vector_search.py       # Atlas Vector Search
│   ├── keyword_search.py      # Atlas BM25 Search
│   ├── hybrid_search.py       # True Reciprocal Rank Fusion (RRF)
│   ├── reranker.py            # Local Cross-Encoder Reranker ($0 cost)
│   └── adaptive_retriever.py  # Intent-aware dynamic routing
│
├── app.py                     # Streamlit web app with Evidence Inspector
├── config.py                  # API key & environment configuration
└── requirements.txt
```


---

# Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Frontend | Streamlit |
| LLM | Google Gemini |
| Agent Framework | LangGraph |
| RAG Framework | LangChain |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector Database | MongoDB Atlas |
| Vector Search | MongoDB Atlas Vector Search |
| Keyword Search | BM25 |
| Rank Fusion | Reciprocal Rank Fusion (RRF) |
| Web Search | Tavily |
| Deployment | Streamlit Community Cloud |

---

# Installation

Clone the repository

```bash
git clone https://github.com/your-username/Hybrid_Search_RAG.git

cd Hybrid_Search_RAG
```

Create virtual environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file.

```env
MONGODB_URI=your_mongodb_connection_string

GOOGLE_API_KEY=your_google_api_key

TAVILY_API_KEY=your_tavily_api_key
```

---

# MongoDB Atlas

Create a Vector Search Index.

Example

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    }
  ]
}
```

Also create a Search Index for keyword search.

---

# Run Application

```bash
streamlit run app.py
```

---

# Agent Workflow

```
START
   │
   ▼
LLM Router
   │
 ┌─┴────────────┐
 │              │
 ▼              ▼
RAG        Web Search
 │              │
 └──────┬───────┘
        ▼
 Generate
        ▼
       END
```

---


# License

This project is licensed under the MIT License.
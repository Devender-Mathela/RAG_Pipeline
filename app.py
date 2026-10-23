import os
import json
import time
import tempfile
import streamlit as st

from ingestion.ingestion import ingest
from agents.graph import graph
from retrieval.adaptive_retriever import classify_query_intent, adaptive_search

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Evidence-Driven RAG Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern enterprise aesthetics
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #1E88E5;
        margin-bottom: 12px;
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-verified { background-color: #E8F5E9; color: #2E7D32; border: 1px solid #A5D6A7; }
    .badge-unverified { background-color: #FFF3E0; color: #E65100; border: 1px solid #FFCC80; }
    .badge-strategy { background-color: #E3F2FD; color: #1565C0; border: 1px solid #90CAF9; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""


# -----------------------------------------------------------------------------
# Sidebar: System Status & Ingestion
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ Agent Control")
    st.caption("Evidence-Driven Knowledge Intelligence System")
    st.divider()

    st.subheader("⚙️ System Architecture")
    st.markdown("""
    - **LLM:** Qwen-2.5-32B (Groq)
    - **Vector DB:** MongoDB Atlas Vector Search
    - **Keyword Engine:** Atlas Search (BM25)
    - **Fusion:** Reciprocal Rank Fusion ($k=60$)
    - **Reranker:** `ms-marco-MiniLM-L-6-v2` (Local)
    - **Web Fallback:** Tavily Search Engine
    """)
    st.divider()

    st.subheader("📄 Document Ingestion")
    uploaded_file = st.file_uploader(
        "Upload reference documents",
        type=["pdf", "csv", "txt", "docx", "html"],
        help="Supported formats: PDF, DOCX, TXT, CSV, HTML"
    )

    if uploaded_file is not None:
        if st.button("📥 Ingest Document", use_container_width=True):
            with st.spinner("Ingesting & generating embeddings..."):
                original_filename = uploaded_file.name
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, original_filename)

                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())

                try:
                    ingest(temp_path)
                    st.success(f"✅ Successfully ingested: {original_filename}")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
                finally:
                    if os.path.exists(temp_dir):
                        import shutil
                        shutil.rmtree(temp_dir)

    st.divider()
    st.caption("Observability & Evidence Inspection Enabled")


# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.title("🛡️ Evidence-Driven Knowledge Intelligence Agent")
st.markdown("Combines **Adaptive Hybrid Retrieval**, **Corrective RAG (CRAG)**, and an **Evidence Verification Guardrail** to eliminate search blind spots and hallucinations.")

# Main Navigation Tabs
tab_assistant, tab_inspector = st.tabs([
    "💬 Knowledge Assistant",
    "🔍 Evidence & Retrieval Inspector"
])



# =============================================================================
# TAB 1: KNOWLEDGE ASSISTANT
# =============================================================================
with tab_assistant:
    st.subheader("Ask a Question")

    # Sample query quick-selector
    sample_queries = [
        "What is the self-attention mechanism in Transformer models?",
        "How does MongoDB Atlas used in RAG?",
        'Find code "ERR_502_BAD_GATEWAY"',
        "What is the secret recipe for Coca-Cola in the Atlanta vault?"
    ]

    selected_sample = st.selectbox(
        "💡 Or pick an example scenario:",
        [""] + sample_queries,
        format_func=lambda x: "Select a sample query..." if x == "" else x
    )

    query_input = st.text_input(
        "Enter your query:",
        value=selected_sample if selected_sample else st.session_state["last_query"],
        placeholder="e.g., Explain self-attention or search for an exact error code..."
    )

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        search_clicked = st.button("🚀 Ask Agent", type="primary", use_container_width=True)

    if search_clicked:
        if not query_input.strip():
            st.warning("Please enter a question.")
        else:
            st.session_state["last_query"] = query_input
            with st.spinner("🔍 Agent analyzing query, executing adaptive retrieval & verifying evidence..."):
                t_start = time.time()
                result = graph.invoke({"question": query_input})
                elapsed = time.time() - t_start
                result["latency_sec"] = round(elapsed, 2)
                st.session_state["last_result"] = result

    # Display Answer & Evidence
    res = st.session_state["last_result"]
    if res:
        st.divider()

        # Evidence Status Banner
        evidence_status = res.get("evidence_status", "UNKNOWN")
        evidence_reason = res.get("evidence_reason", "")

        if evidence_status == "VERIFIED":
            st.success(f"🛡️ **Evidence Verified:** {evidence_reason}")
        elif evidence_status in ["UNVERIFIED", "INSUFFICIENT_EVIDENCE"]:
            st.warning(f"⚠️ **Evidence Notice:** {evidence_reason}")
        else:
            st.info("ℹ️ Answer generated without specific evidence warnings.")

        # Query Self-Correction Banner
        if res.get("retry_count", 0) > 0:
            st.info(f"🔄 **Query Reformulated (Self-Correction):** The initial search was insufficient. Automatically reformulated to: *'{res.get('question')}'*")

        # Answer Display Card
        st.subheader("💡 Answer")
        st.write(res.get("answer", ""))

        # Metadata Tags
        tag_cols = st.columns([2, 2, 2, 2])
        with tag_cols[0]:
            st.metric("Route Taken", res.get("route", "").upper())
        with tag_cols[1]:
            st.metric("Retrieval Strategy", res.get("search_strategy", "N/A").upper())
        with tag_cols[2]:
            doc_rel = res.get("documents_relevant")
            rel_label = "Relevant" if doc_rel else ("Low Relevance" if doc_rel is False else "Web / N/A")
            st.metric("Context Quality", rel_label)
        with tag_cols[3]:
            st.metric("End-to-End Latency", f"{res.get('latency_sec', 0.0)}s")

        st.divider()

        # Sources Accordion
        docs = res.get("documents", [])
        if res.get("route") == "rag":
            st.subheader(f"📖 Retrieved Sources ({len(docs)} Chunks)")
            if not docs:
                st.info("No documents retrieved from internal repository.")
            else:
                for idx, doc in enumerate(docs, start=1):
                    meta = doc.get("metadata", {})
                    source_name = os.path.basename(meta.get("source", "Document Chunk"))
                    page = meta.get("page")
                    header_label = f"[{idx}] {source_name}" + (f" | Page {page + 1}" if page is not None else "")

                    with st.expander(header_label):
                        st.write(doc.get("text", ""))

                        score_badges = []
                        if "rerank_score" in doc:
                            score_badges.append(f"Cross-Encoder Score: **{doc['rerank_score']:.4f}**")
                        if "rrf_score" in doc:
                            score_badges.append(f"RRF Score: **{doc['rrf_score']:.5f}**")
                        if "score" in doc:
                            score_badges.append(f"Base Score: **{doc['score']:.4f}**")

                        if score_badges:
                            st.caption(" • ".join(score_badges))
        else:
            st.subheader(f"🌐 Web Fallback Sources ({len(docs)} Results)")
            for idx, doc in enumerate(docs, start=1):
                meta = doc.get("metadata", {})
                title = meta.get("title", "Web Result")
                url = meta.get("source", "")
                with st.expander(f"[{idx}] {title}"):
                    st.write(doc.get("text", ""))
                    if url:
                        st.markdown(f"**URL:** [{url}]({url})")


# =============================================================================
# TAB 2: EVIDENCE & RETRIEVAL INSPECTOR
# =============================================================================
with tab_inspector:
    st.subheader("🔍 Deep Observability & Internal State")
    st.markdown("Examine the decision-making process, retrieval strategy classification, and evidence guardrail verification.")

    res = st.session_state["last_result"]
    if not res:
        st.info("Ask a question in the **Knowledge Assistant** tab first to inspect runtime agent telemetry.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 1. Adaptive Query Routing")
            # Analyze intent on the query
            intent_analysis = classify_query_intent(res.get("original_question", res.get("question", "")))
            st.markdown(f"- **Original Query:** `{res.get('original_question', res.get('question'))}`")
            st.markdown(f"- **Strategy Decided:** `{intent_analysis['strategy'].upper()}`")
            st.markdown(f"- **Classification Reason:** {intent_analysis['reason']}")
            st.markdown(f"- **Router Target:** `{res.get('route', '').upper()}`")

            st.markdown("### 2. Document Relevance Grading (CRAG)")
            doc_grade = res.get("documents_relevant")
            if doc_grade is True:
                st.success("✅ **Passed:** Retrieved passages contained adequate semantic evidence.")
            elif doc_grade is False:
                st.warning("⚠️ **Failed:** Initial retrieval was insufficient or irrelevant.")
            else:
                st.info("ℹ️ External web route bypasses document grading.")

            st.markdown(f"- **Rewrite Retries Triggered:** `{res.get('retry_count', 0)} / 1`")

        with col2:
            st.markdown("### 3. Evidence Verification Guardrail")
            ev_status = res.get("evidence_status", "UNKNOWN")
            if ev_status == "VERIFIED":
                st.success(f"🛡️ **Status:** `VERIFIED`")
            else:
                st.warning(f"⚠️ **Status:** `{ev_status}`")

            st.markdown(f"- **Verifier Evaluation:** {res.get('evidence_reason', 'No evaluation available.')}")

            st.markdown("### 4. Candidate Reranking Breakdown")
            docs = res.get("documents", [])
            if docs:
                table_data = []
                for idx, doc in enumerate(docs, start=1):
                    table_data.append({
                        "Rank": idx,
                        "Source": os.path.basename(doc.get("metadata", {}).get("source", "Chunk")),
                        "Cross-Encoder Score": doc.get("rerank_score", "N/A"),
                        "RRF Score": doc.get("rrf_score", "N/A"),
                        "Text Preview": doc.get("text", "")[:70] + "..."
                    })
                st.dataframe(table_data, use_container_width=True)
            else:
                st.caption("No internal candidate chunks were reranked.")
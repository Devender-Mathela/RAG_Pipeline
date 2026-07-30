import os
import tempfile

import streamlit as st

from ingestion.ingestion import ingest
from agents.graph import graph


st.set_page_config(
    page_title="Hybrid Search RAG",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Agentic Hybrid RAG")
st.write("Upload a PDF and ask questions. The agent will decide whether to answer from your documents or perform a web search.")

# -------------------------------------------------
# Upload PDF
# -------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a document",
    type=[
        "pdf",
        "csv",
        "txt",
        "docx",
        "html"
    ]
)

if uploaded_file is not None:

    if st.button("Ingest Document"):

        with st.spinner("Ingesting document..."):

            original_filename = uploaded_file.name
            
            import shutil
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, original_filename)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            try:

                ingest(temp_path)

                st.success("✅ Document ingested successfully!")

            except Exception as e:

                st.error(str(e))

            finally:

                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

st.divider()

# -------------------------------------------------
# Ask Question
# -------------------------------------------------

question = st.text_input(
    "Ask a question"
)

if st.button("Search"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Thinking..."):

            result = graph.invoke(
                {
                    "question": question
                }
            )

            st.subheader("💡 Answer")

            st.write(result["answer"])

            st.caption(
                f"Route Selected: {result['route'].upper()}"
            )

            st.divider()

            # ---------------------------------------
            # RAG Sources
            # ---------------------------------------

            if result["route"] == "rag":

                st.subheader("📖 Sources")

                if not result["documents"]:

                    st.info("No documents retrieved.")

                else:

                    for index, doc in enumerate(
                        result["documents"],
                        start=1
                    ):

                        metadata = doc.get("metadata", {})

                        title = metadata.get(
                            "title",
                            ""
                        )

                        source = os.path.basename(
                            metadata.get(
                                "source",
                                "Unknown Document"
                            )
                        )

                        page = metadata.get("page")

                        if page is not None:
                            page += 1

                        header = (
                            title if title else source
                        )
                        
                        header_text = f"[{index}] {header}"
                        if page is not None:
                            header_text += f" | Page {page}"

                        with st.expander(header_text):

                            st.write(doc["text"])

                            if "score" in doc:

                                st.caption(
                                    f"Similarity Score: {doc['score']:.4f}"
                                )

            # ---------------------------------------
            # Web Sources
            # ---------------------------------------

            else:

                st.subheader("🌐 Web Sources")

                for index, doc in enumerate(
                    result["documents"],
                    start=1
                ):

                    metadata = doc.get("metadata", {})

                    title = metadata.get(
                        "title",
                        "Web Source"
                    )

                    source = metadata.get(
                        "source",
                        ""
                    )

                    with st.expander(
                        f"[{index}] {title}"
                    ):

                        st.write(doc["text"])

                        if source:
                            st.markdown(
                                f"**URL:** {source}"
                            )
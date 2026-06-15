"""
Document Research Workbench

Current features:
- Multi-document upload
- MarkItDown parsing
- Markdown viewer
- Chunking
- Chunk viewer
- Gemini embeddings
- FAISS vector store
- Retrieval search
- Retrieved chunks viewer
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv

from ingestion.parser import convert_to_markdown
from ingestion.chunker import chunk_markdown_text
from ingestion.embeddings import get_embedding_model

from vectorstore.faiss_store import (
    create_documents_from_chunks,
    create_faiss_index,
    add_to_faiss_index,
)

from retrieval.retriever import retrieve_chunks

from llm.gemini import (
    build_prompt,
    ask_gemini
)

from utils.chat_history import (
    format_conversation_history,
    trim_history
)

from utils.costs import estimate_gemini_cost

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="Document Research Workbench",
    layout="wide",
)

st.title("📚 Document Research Workbench")


# --------------------------------------------------
# Constants
# --------------------------------------------------

TOP_K = 4


# --------------------------------------------------
# Create uploads folder if it does not exist
# --------------------------------------------------

if not os.path.exists("uploads"):
    os.makedirs("uploads")


# --------------------------------------------------
# Create embedding model
# --------------------------------------------------
# This is used for both:
# 1. Embedding document chunks
# 2. Embedding the user's search question

embedding_model = get_embedding_model()


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "documents" not in st.session_state:
    st.session_state.documents = []

if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "session_prompt_tokens" not in st.session_state:
    st.session_state.session_prompt_tokens = 0

if "session_completion_tokens" not in st.session_state:
    st.session_state.session_completion_tokens = 0

if "session_total_tokens" not in st.session_state:
    st.session_state.session_total_tokens = 0

if "session_cost" not in st.session_state:
    st.session_state.session_cost = 0


# --------------------------------------------------
# Sidebar Workspace
# --------------------------------------------------

st.sidebar.header("Workspace")

st.sidebar.write(f"Documents loaded: {len(st.session_state.documents)}")
st.sidebar.write(f"Total chunks: {len(st.session_state.all_chunks)}")

if st.session_state.vector_store is not None:
    st.sidebar.success("FAISS Index Ready")
else:
    st.sidebar.info("FAISS Index Not Created Yet")

if st.sidebar.button("Clear Session"):
    st.session_state.documents = []
    st.session_state.all_chunks = []
    st.session_state.vector_store = None
    st.success("Session cleared.")

if st.session_state.documents:
    st.sidebar.subheader("Loaded Documents")

    for doc in st.session_state.documents:
        st.sidebar.write(
            f"- {doc['file_name']} ({doc['chunk_count']} chunks)"
        )

st.sidebar.markdown("---")

st.sidebar.subheader(
    "Session Metrics"
)

st.sidebar.write(
    f"Prompt Tokens: {st.session_state.session_prompt_tokens:,}"
)

st.sidebar.write(
    f"Completion Tokens: {st.session_state.session_completion_tokens:,}"
)

st.sidebar.write(
    f"Total Tokens: {st.session_state.session_total_tokens:,}"
)

st.sidebar.write(
    f"Cost: ${st.session_state.session_cost:.6f}"
)

st.sidebar.subheader(
    "Current Settings"
)

st.sidebar.write("TOP_K = 4")
st.sidebar.write("MAX_HISTORY = 7")
st.sidebar.write("Chunk Size = 1200")
st.sidebar.write("Chunk Overlap = 200")

# --------------------------------------------------
# Upload Files
# --------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload one or more documents",
    type=[
        "pdf",
        "docx",
        "pptx",
        "xlsx",
        "txt",
        "html",
        "md",
    ],
    accept_multiple_files=True,
)


# --------------------------------------------------
# Process Uploaded Files
# --------------------------------------------------

if uploaded_files:

    for uploaded_file in uploaded_files:

        # Check if file already exists in this session
        already_loaded = any(
            doc["file_name"] == uploaded_file.name
            for doc in st.session_state.documents
        )

        if already_loaded:
            st.warning(f"{uploaded_file.name} is already loaded. Skipping.")
            continue

        # Save uploaded file locally
        file_path = os.path.join("uploads", uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"Uploaded: {uploaded_file.name}")


        # --------------------------------------------------
        # Step 1: Convert document to markdown
        # --------------------------------------------------

        markdown_start_time = time.time()

        markdown_text = convert_to_markdown(file_path)

        markdown_time = time.time() - markdown_start_time

        if not markdown_text or not markdown_text.strip():
            st.error(f"No readable markdown found in {uploaded_file.name}. Skipping.")
            continue

        st.success(
            f"Converted {uploaded_file.name} to markdown in {markdown_time:.2f}s"
        )


        # --------------------------------------------------
        # Step 2: Chunk markdown
        # --------------------------------------------------

        chunk_start_time = time.time()

        chunks = chunk_markdown_text(
            markdown_text=markdown_text,
            source_file=uploaded_file.name,
        )

        chunk_time = time.time() - chunk_start_time

        if len(chunks) == 0:
            st.error(f"No usable chunks found in {uploaded_file.name}. Skipping.")
            continue

        st.success(
            f"Created {len(chunks)} chunks in {chunk_time:.2f}s"
        )


        # --------------------------------------------------
        # Step 3: Convert chunks into LangChain Documents
        # --------------------------------------------------

        documents = create_documents_from_chunks(chunks)

        if len(documents) == 0:
            st.error(f"No LangChain documents created for {uploaded_file.name}. Skipping.")
            continue


        # --------------------------------------------------
        # Step 4: Embed and store in FAISS
        # --------------------------------------------------

        embedding_start_time = time.time()

        try:
            if st.session_state.vector_store is None:
                st.session_state.vector_store = create_faiss_index(
                    documents,
                    embedding_model,
                )

                st.success("Created new FAISS index.")

            else:
                st.session_state.vector_store = add_to_faiss_index(
                    st.session_state.vector_store,
                    documents,
                )

                st.success("Added document to existing FAISS index.")

        except Exception as e:
            st.error(f"Embedding or FAISS error: {e}")
            continue

        embedding_time = time.time() - embedding_start_time

        st.success(
            f"Embeddings created and stored in {embedding_time:.2f}s"
        )


        # --------------------------------------------------
        # Step 5: Save document info in session
        # --------------------------------------------------

        st.session_state.documents.append(
            {
                "file_name": uploaded_file.name,
                "markdown": markdown_text,
                "chunk_count": len(chunks),
                "markdown_time": markdown_time,
                "chunk_time": chunk_time,
                "embedding_time": embedding_time,
            }
        )

        st.session_state.all_chunks.extend(chunks)


# --------------------------------------------------
# Search / Retrieval Section
# --------------------------------------------------

st.header("Ask Questions")

question = st.text_input(
    "Ask a question about the uploaded documents"
)

search_clicked = st.button(
    "Ask Gemini"
)

if search_clicked:

    cleaned_question = question.strip() if question else ""

    if not cleaned_question:

        st.warning(
            "Please enter a question."
        )

    elif st.session_state.vector_store is None:

        st.warning(
            "Please upload documents first."
        )

    else:

        total_start_time = time.time()

        # ---------------------------------------
        # Retrieval
        # ---------------------------------------

        retrieval_start = time.time()

        results = retrieve_chunks(
            vector_store=st.session_state.vector_store,
            question=cleaned_question,
            top_k=4
        )

        retrieval_time = (
            time.time()
            - retrieval_start
        )

        if len(results) == 0:

            st.warning(
                "No chunks retrieved."
            )

        else:

            # ---------------------------------------
            # Display Retrieved Chunks
            # ---------------------------------------

            with st.expander(
                "Retrieved Chunks",
                expanded=True
            ):

                for rank, (doc, score) in enumerate(results):

                    st.markdown("---")

                    st.write(
                        f"### Result {rank+1}"
                    )

                    st.write(
                        f"Source: {doc.metadata.get('source_file')}"
                    )

                    st.write(
                        f"Chunk ID: {doc.metadata.get('chunk_id')}"
                    )

                    st.write(
                        f"Distance Score: {score}"
                    )

                    st.text_area(
                        f"Chunk {rank+1}",
                        doc.page_content,
                        height=200
                    )

            # ---------------------------------------
            # Build Conversation History
            # ---------------------------------------

            trimmed_history = trim_history(
                st.session_state.chat_messages,
                max_messages=7
            )

            history_text = (
                format_conversation_history(
                    trimmed_history
                )
            )

            # ---------------------------------------
            # Build Prompt
            # ---------------------------------------

            prompt_start = time.time()

            retrieved_docs = [
                doc
                for doc, score
                in results
            ]

            prompt = build_prompt(
                question=cleaned_question,
                retrieved_chunks=retrieved_docs,
                conversation_history=history_text
            )

            prompt_time = (
                time.time()
                - prompt_start
            )

            # ---------------------------------------
            # Show Prompt
            # ---------------------------------------

            with st.expander(
                "Prompt Sent To Gemini",
                expanded=False
            ):
                st.text_area(
                    "Prompt",
                    prompt,
                    height=500
                )

            # ---------------------------------------
            # Gemini Call
            # ---------------------------------------

            llm_start = time.time()

            answer, usage = ask_gemini(
                prompt
            )

            citations = []

            for doc in retrieved_docs:

                citation = {
                    "source_file": doc.metadata.get(
                        "source_file",
                        "Unknown"
                    ),
                    "chunk_id": doc.metadata.get(
                        "chunk_id",
                        "Unknown"
                    ),
                    "content": doc.page_content
                }

                citations.append(citation)

            prompt_tokens = usage.get(
                "prompt_tokens",
                0
            )

            completion_tokens = usage.get(
                "completion_tokens",
                0
            )

            total_tokens = usage.get(
                "total_tokens",
                0
            )

            request_cost = estimate_gemini_cost(
                prompt_tokens,
                completion_tokens
            )

            st.session_state.session_prompt_tokens += (
                prompt_tokens
            )

            st.session_state.session_completion_tokens += (
                completion_tokens
            )

            st.session_state.session_total_tokens += (
                total_tokens
            )

            st.session_state.session_cost += (
                request_cost
            )

            with st.expander(
                "Token Usage",
                expanded=False
            ):

                st.write(
                    f"Prompt Tokens: {prompt_tokens:,}"
                )

                st.write(
                    f"Completion Tokens: {completion_tokens:,}"
                )

                st.write(
                    f"Total Tokens: {total_tokens:,}"
                )

                st.write(
                    f"Estimated Cost: ${request_cost:.6f}"
                )

            llm_time = (
                time.time()
                - llm_start
            )

            # ---------------------------------------
            # Save History
            # ---------------------------------------

            st.session_state.chat_messages.append(
                {
                    "role": "User",
                    "content": cleaned_question
                }
            )

            st.session_state.chat_messages.append(
                {
                    "role": "Assistant",
                    "content": answer
                }
            )

            # ---------------------------------------
            # Answer
            # ---------------------------------------

            st.subheader(
                "Gemini Response"
            )

            st.write(
                answer
            )

            st.markdown("### Sources")
            unique_sources = set()

            for citation in citations:

                source = citation["source_file"]

                if source not in unique_sources:

                    unique_sources.add(source)

                    st.write(
                        f"• {source}"
                    )

            # Citation Viewer
            with st.expander(
                "Citations",
                expanded=False
            ):

                for i, citation in enumerate(citations):

                    st.markdown("---")

                    st.write(
                        f"### Citation {i+1}"
                    )

                    st.write(
                        f"Source File: {citation['source_file']}"
                    )

                    st.write(
                        f"Chunk ID: {citation['chunk_id']}"
                    )

                    st.text_area(
                        f"Citation Chunk {i+1}",
                        citation["content"],
                        height=250
                    )
            

            total_time = (
                time.time()
                - total_start_time
            )

            # ---------------------------------------
            # Timing Metrics
            # ---------------------------------------

            with st.expander(
                "Timing Metrics"
            ):

                st.write(
                    f"Retrieval Time: {retrieval_time:.2f}s"
                )

                st.write(
                    f"Prompt Build Time: {prompt_time:.2f}s"
                )

                st.write(
                    f"LLM Time: {llm_time:.2f}s"
                )

                st.write(
                    f"Total Time: {total_time:.2f}s"
                )


# --------------------------------------------------
# Display Markdown Outputs
# --------------------------------------------------

if st.session_state.documents:

    st.header("Document Outputs")

    for doc in st.session_state.documents:

        with st.expander(f"Markdown Output - {doc['file_name']}"):

            st.write("Processing report:")

            st.write(
                f"""
                Markdown Time: {doc['markdown_time']:.2f}s

                Chunking Time: {doc['chunk_time']:.2f}s

                Embedding Time: {doc['embedding_time']:.2f}s

                Chunks Created: {doc['chunk_count']}
                """
            )

            st.text_area(
                label=f"Markdown for {doc['file_name']}",
                value=doc["markdown"],
                height=400,
            )


# --------------------------------------------------
# Display Generated Chunks
# --------------------------------------------------

if st.session_state.all_chunks:

    with st.expander("Generated Chunks", expanded=False):

        st.write(f"Total chunks: {len(st.session_state.all_chunks)}")

        for chunk in st.session_state.all_chunks:

            st.markdown("---")

            st.write(f"**Chunk ID:** {chunk['chunk_id']}")
            st.write(f"**Source File:** {chunk['source_file']}")
            st.write(f"**Character Count:** {chunk['char_count']}")

            st.text_area(
                label=f"Chunk {chunk['chunk_id']}",
                value=chunk["text"],
                height=200,
            )


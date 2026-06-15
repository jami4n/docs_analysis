"""
faiss_store.py

Handles creation and updating
of the FAISS vector database.
"""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def create_documents_from_chunks(chunks):
    """
    Convert chunk dictionaries into LangChain Documents.
    """

    docs = []

    for chunk in chunks:

        doc = Document(
            page_content=chunk["text"],
            metadata={
                "chunk_id": chunk["chunk_id"],
                "source_file": chunk["source_file"]
            }
        )

        docs.append(doc)

    return docs


def create_faiss_index(
    documents,
    embedding_model
):
    """
    Create a brand new FAISS index.
    """

    return FAISS.from_documents(
        documents,
        embedding_model
    )


def add_to_faiss_index(
    faiss_index,
    documents
):
    """
    Add new documents to existing index.
    """

    faiss_index.add_documents(documents)

    return faiss_index
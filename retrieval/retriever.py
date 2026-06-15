"""
retriever.py

Handles vector search against FAISS.
"""


def retrieve_chunks(
    vector_store,
    question,
    top_k=4
):
    """
    Retrieve the most relevant chunks.

    Parameters:
        vector_store:
            FAISS index

        question:
            User question

        top_k:
            Number of chunks to retrieve

    Returns:
        List of retrieved documents
    """

    results = vector_store.similarity_search_with_score(
        question,
        k=top_k
    )

    return results
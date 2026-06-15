"""
embeddings.py

Creates the Google embedding model.
This model converts text chunks into vectors for FAISS.
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings


def get_embedding_model():
    """
    Return the embedding model.

    We use gemini-embedding-001 because text-embedding-004
    may not be available for your Gemini API version/account.
    """

    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )
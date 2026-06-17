"""
embeddings.py

Creates Google embedding model for FAISS.
Works locally with .env and on Streamlit Cloud with st.secrets.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def get_google_api_key():
    """
    Get API key from either:
    1. Streamlit Cloud secrets
    2. Local .env file
    """

    load_dotenv()

    # Try Streamlit secrets first
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass

    # Fallback to local .env
    return os.getenv("GOOGLE_API_KEY")


def get_embedding_model():
    """
    Create the Gemini embedding model.
    """

    api_key = get_google_api_key()

    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in Streamlit secrets or .env")

    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )
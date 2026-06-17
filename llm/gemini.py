"""
gemini.py

Handles:
1. Gemini client setup
2. Prompt building
3. Calling Gemini Flash

Works both:
- Locally (.env)
- Streamlit Cloud (st.secrets)
"""

import os
import streamlit as st

from dotenv import load_dotenv
from google import genai


# --------------------------------------------------
# API KEY HELPER
# --------------------------------------------------

def get_google_api_key():
    """
    Try Streamlit secrets first,
    then fallback to .env
    """

    load_dotenv()

    try:
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass

    return os.getenv("GOOGLE_API_KEY")


# --------------------------------------------------
# GEMINI CLIENT
# --------------------------------------------------

def get_gemini_client():

    api_key = get_google_api_key()

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found in Streamlit secrets or .env"
        )

    client = genai.Client(
        api_key=api_key
    )

    return client


# --------------------------------------------------
# BUILD PROMPT
# --------------------------------------------------

def build_prompt(
    question,
    retrieved_chunks,
    conversation_history=""
):
    """
    Build prompt sent to Gemini.

    Gemini should answer only
    using retrieved document chunks.
    """

    context_text = ""

    for i, chunk in enumerate(retrieved_chunks, start=1):

        context_text += f"""

CHUNK {i}

SOURCE FILE:
{chunk.metadata.get("source_file", "Unknown")}

CHUNK ID:
{chunk.metadata.get("chunk_id", "Unknown")}

CONTENT:
{chunk.page_content}

"""

    prompt = f"""
You are a document research assistant.

Use ONLY the information contained in the retrieved context.

You may:
- Summarize information
- Explain information
- Reorganize information
- Combine findings across chunks

You must NOT:
- Use outside knowledge
- Invent facts
- Guess

If the answer cannot be found in the context, respond:

"The uploaded documents do not contain enough information to answer this question."

At the end of your answer include:

Sources:
- Source File
- Chunk ID

--------------------------------------------------

RETRIEVED CONTEXT

{context_text}

--------------------------------------------------

CONVERSATION HISTORY

{conversation_history}

--------------------------------------------------

QUESTION

{question}
"""

    return prompt


# --------------------------------------------------
# ASK GEMINI
# --------------------------------------------------

def ask_gemini(prompt):
    """
    Send prompt to Gemini Flash.

    Returns:
        answer
        usage stats
    """

    client = get_gemini_client()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    answer = response.text

    usage = {}

    if hasattr(response, "usage_metadata"):

        usage = {
            "prompt_tokens": getattr(
                response.usage_metadata,
                "prompt_token_count",
                0
            ),
            "completion_tokens": getattr(
                response.usage_metadata,
                "candidates_token_count",
                0
            ),
            "total_tokens": getattr(
                response.usage_metadata,
                "total_token_count",
                0
            )
        }

    return answer, usage
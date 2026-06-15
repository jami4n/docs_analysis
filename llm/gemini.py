"""
gemini.py

Handles:
1. Gemini client setup
2. Prompt building
3. Calling Gemini Flash
"""

import os
from dotenv import load_dotenv
from google import genai


def get_gemini_client():
    """
    Creates the Gemini client using the GOOGLE_API_KEY
    stored in your .env file.
    """

    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Check your .env file.")

    client = genai.Client(api_key=api_key)

    return client


def build_prompt(question, retrieved_chunks, conversation_history=""):
    """
    Builds the prompt that will be sent to Gemini.

    Gemini should only use retrieved document chunks
    as its source of facts.
    """

    context_text = ""

    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_text += f"""
CHUNK {i}
SOURCE FILE: {chunk.metadata.get("source_file", "Unknown")}
CHUNK ID: {chunk.metadata.get("chunk_id", "Unknown")}

{chunk.page_content}

"""

    prompt = f"""
You are a document research assistant.

Use ONLY the information contained in the retrieved context.

You may:
- Summarize information
- Explain information in simple language
- Reorganize information
- Combine findings across chunks

You must NOT:
- Use outside knowledge
- Invent facts
- Guess
- Add information that is not supported by the retrieved context

If the answer cannot be found in the retrieved context, respond:

"The uploaded documents do not contain enough information to answer this question."

Always include a short "Sources" section at the end using the source file and chunk ID.

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


def ask_gemini(prompt):
    """
    Sends prompt to Gemini and returns:

    - answer
    - usage metadata
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
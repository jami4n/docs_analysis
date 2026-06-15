"""
chunker.py

Splits markdown text into clean chunks.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_markdown_text(markdown_text, source_file):
    """
    Splits markdown text into smaller chunks and removes empty chunks.
    """

    chunk_size = 1200
    chunk_overlap = 200

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    texts = splitter.split_text(markdown_text)

    chunks = []

    for index, text in enumerate(texts):

        # Clean the text
        cleaned_text = text.strip()

        # Skip empty chunks
        if not cleaned_text:
            continue

        chunks.append(
            {
                "chunk_id": f"{source_file}_chunk_{len(chunks) + 1}",
                "source_file": source_file,
                "text": cleaned_text,
                "char_count": len(cleaned_text)
            }
        )

    return chunks
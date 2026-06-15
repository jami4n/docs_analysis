"""
parser.py

Responsible for converting uploaded files
into Markdown using MarkItDown.
"""

from markitdown import MarkItDown


def convert_to_markdown(file_path):
    """
    Converts a file into markdown.
    """

    try:
        md = MarkItDown()

        result = md.convert(file_path)

        return result.text_content

    except Exception as e:
        print(f"Error converting file: {e}")

        return None
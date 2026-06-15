"""
chat_history.py

Handles conversation history.

For V1 we only keep the last
7 messages.
"""


def format_conversation_history(messages):
    """
    Convert message list into text.

    Example:

    User: Hello
    Assistant: Hi
    """

    history = ""

    for msg in messages:

        role = msg["role"]
        content = msg["content"]

        history += f"{role}: {content}\n"

    return history


def trim_history(
    messages,
    max_messages=7
):
    """
    Keep only the last N messages.
    """

    return messages[-max_messages:]
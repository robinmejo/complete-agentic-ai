from .llm import llm
from .message_utils import message_content_to_text


# ------------------------------------------------------------
# Clean generated title
# ------------------------------------------------------------

def clean_title(title):

    if not title:
        return "New Conversation"

    title = message_content_to_text(
        title
    )

    title = title.strip()

    # Remove quotation marks
    title = title.strip('"').strip("'")

    # Remove accidental markdown heading
    title = title.lstrip("#").strip()

    # Keep only first line
    lines = title.splitlines()

    if lines:
        title = lines[0].strip()

    # Limit title length
    if len(title) > 45:
        title = (
            title[:42].rstrip()
            + "..."
        )

    return title or "New Conversation"


# ------------------------------------------------------------
# Generate title using LLM
# ------------------------------------------------------------

def generate_chat_title(user_message):

    prompt = f"""
Generate a clear and meaningful title for the conversation below.

Rules:
- Use between 3 and 6 words
- Describe the main topic
- Do not use quotation marks
- Do not use a full stop
- Do not prefix the title with "Title"
- Return only the title and nothing else

First user message:
{user_message}
""".strip()

    response = llm.invoke(
        prompt
    )

    title = message_content_to_text(
        response.content
    )

    return clean_title(
        title
    )


# ------------------------------------------------------------
# Fallback title
# ------------------------------------------------------------

def create_fallback_title(user_message):

    user_message = message_content_to_text(
        user_message
    )

    clean_message = " ".join(
        user_message.split()
    )

    if not clean_message:
        return "New Conversation"

    words = clean_message.split()

    fallback_title = " ".join(
        words[:6]
    )

    if len(words) > 6:
        fallback_title += "..."

    return clean_title(
        fallback_title
    )
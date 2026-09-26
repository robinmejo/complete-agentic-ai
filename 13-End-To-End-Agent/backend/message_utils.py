# ------------------------------------------------------------
# Convert Cohere content to plain text
# ------------------------------------------------------------

def cohere_text_only(content):

    if isinstance(content, str):
        return content

    if isinstance(content, dict):

        if (
            content.get("type") == "text"
            and isinstance(content.get("text"), str)
        ):
            return content["text"]

        nested_content = content.get("content")

        if nested_content is not None:
            return cohere_text_only(
                nested_content
            )

        return ""

    if isinstance(content, list):

        parts = [
            cohere_text_only(item)
            for item in content
        ]

        return "".join(
            part
            for part in parts
            if part
        )

    return ""


# ------------------------------------------------------------
# Convert any message content to displayable text
# ------------------------------------------------------------

def message_content_to_text(content):

    return cohere_text_only(content)
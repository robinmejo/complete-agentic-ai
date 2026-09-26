from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI

from .config import (
    COHERE_API_KEY,
    OPENAI_API_KEY,
    USE_OPENAI,
)


# ------------------------------------------------------------
# Create selected LLM
# ------------------------------------------------------------

def create_llm():

    if USE_OPENAI:

        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY was not found. "
                "Add OPENAI_API_KEY to your .env file."
            )

        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
        )

    else:

        if not COHERE_API_KEY:
            raise ValueError(
                "COHERE_API_KEY was not found. "
                "Add COHERE_API_KEY to your .env file."
            )

        return ChatCohere(
            model="command-a-plus-05-2026",
            cohere_api_key=COHERE_API_KEY,
        )


# ------------------------------------------------------------
# Shared LLM instance
# ------------------------------------------------------------

llm = create_llm()
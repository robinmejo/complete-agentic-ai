import os
import sqlite3
from pathlib import Path
from threading import RLock
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


# ------------------------------------------------------------
# Environment configuration
# ------------------------------------------------------------

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set True to use OpenAI.
# Set False to use Cohere.
use_openai = False


# ------------------------------------------------------------
# Initialize selected language model
# ------------------------------------------------------------

if use_openai:
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY was not found. "
            "Add OPENAI_API_KEY to your .env file."
        )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
    )

else:
    if not COHERE_API_KEY:
        raise ValueError(
            "COHERE_API_KEY was not found. "
            "Add COHERE_API_KEY to your .env file."
        )

    llm = ChatCohere(
        model="command-a-plus-05-2026",
        cohere_api_key=COHERE_API_KEY,
    )


# ------------------------------------------------------------
# LangGraph state
# ------------------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ------------------------------------------------------------
# Message utility
# ------------------------------------------------------------

def cohere_text_only(content):
    """
    Convert Cohere message content into plain text.

    Cohere content may be:
    - A string
    - A dictionary
    - A list
    - A nested content structure
    """

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
            return cohere_text_only(nested_content)

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
# LangGraph chatbot node
# ------------------------------------------------------------

def chat_node(state: ChatState):
    """
    Invoke the selected LLM with the complete conversation history.
    """

    messages = state["messages"]

    # Cohere can save content as lists or dictionaries.
    # Normalize HumanMessage and AIMessage content to strings.

    if not use_openai:
        normalized_messages = []

        for message in messages:

            if isinstance(message, HumanMessage):
                normalized_messages.append(
                    HumanMessage(
                        content=cohere_text_only(
                            message.content
                        )
                    )
                )

            elif isinstance(message, AIMessage):
                normalized_messages.append(
                    AIMessage(
                        content=cohere_text_only(
                            message.content
                        )
                    )
                )

            else:
                normalized_messages.append(message)

        messages = normalized_messages

    response = llm.invoke(messages)

    return {
        "messages": [response]
    }


# ------------------------------------------------------------
# SQLite database
# ------------------------------------------------------------

# Use a fixed path relative to this Python file.
# This prevents creating different chatbot.db files when the
# application is started from different working directories.

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "chatbot.db"

conn = sqlite3.connect(
    database=str(DATABASE_PATH),
    check_same_thread=False,
    timeout=30,
)

# Allow SQLite to wait for a locked database instead of failing
# immediately.

conn.execute("PRAGMA busy_timeout = 30000")

# Protect custom metadata database operations.
database_lock = RLock()

# LangGraph checkpoint storage.
checkpoint = SqliteSaver(conn)


# Store conversation titles separately from LangGraph checkpoints.
with database_lock:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_titles (
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

conn.commit()


# ------------------------------------------------------------
# Build LangGraph
# ------------------------------------------------------------

graph = StateGraph(ChatState)

graph.add_node(
    "chat_node",
    chat_node,
)

graph.add_edge(
    START,
    "chat_node",
)

graph.add_edge(
    "chat_node",
    END,
)

chatbot = graph.compile(
    checkpointer=checkpoint
)


# ------------------------------------------------------------
# Conversation title database operations
# ------------------------------------------------------------

def save_thread_title(thread_id, title):
    """
    Insert a conversation title or update the title of an
    existing conversation.
    """

    thread_id = str(thread_id).strip()
    title = str(title).strip()

    if not thread_id:
        raise ValueError(
            "thread_id cannot be empty"
        )

    if not title:
        title = "New Conversation"

    with database_lock:
        conn.execute(
            """
            INSERT INTO conversation_titles (
                thread_id,
                title,
                created_at,
                updated_at
            )
            VALUES (
                ?,
                ?,
                CURRENT_TIMESTAMP,
                CURRENT_TIMESTAMP
            )
            ON CONFLICT(thread_id)
            DO UPDATE SET
                title = excluded.title,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                thread_id,
                title,
            )
        )

        conn.commit()


def get_saved_thread_titles():
    """
    Return the saved titles as a dictionary.

    Example:

    {
        "thread-id-1": "Python String Methods",
        "thread-id-2": "LangGraph SQLite Memory"
    }
    """

    with database_lock:
        rows = conn.execute(
            """
            SELECT
                thread_id,
                title
            FROM conversation_titles
            """
        ).fetchall()

    return {
        row[0]: row[1]
        for row in rows
    }


def get_all_threads_with_titles():
    """
    Return LangGraph thread IDs with their saved titles.

    LangGraph can have several checkpoints for one thread.
    The dictionary prevents duplicate thread IDs.
    """

    saved_titles = get_saved_thread_titles()

    conversations = {}

    for ckpt in checkpoint.list(None):

        configurable = ckpt.config.get(
            "configurable",
            {}
        )

        thread_id = configurable.get(
            "thread_id"
        )

        if not thread_id:
            continue

        thread_id = str(thread_id)

        if thread_id not in conversations:
            conversations[thread_id] = (
                saved_titles.get(
                    thread_id,
                    "New Conversation"
                )
            )

    return conversations


def delete_thread(thread_id):
    """
    Delete a conversation completely.

    This removes:
    1. LangGraph checkpoints and pending writes
    2. The title from conversation_titles
    """

    thread_id = str(thread_id).strip()

    if not thread_id:
        raise ValueError(
            "thread_id cannot be empty"
        )

    with database_lock:

        # Delete LangGraph checkpoints and writes.
        checkpoint.delete_thread(thread_id)

        # Delete the conversation title.
        conn.execute(
            """
            DELETE FROM conversation_titles
            WHERE thread_id = ?
            """,
            (thread_id,)
        )

        conn.commit()

    return True
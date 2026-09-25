import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

from agentic_chatbot_backend_new import (
    chatbot,
    llm,
    use_openai,
    get_all_threads_with_titles,
    save_thread_title,
    delete_thread,
)

# ------------------------------------------------------------
# Streamlit configuration
# ------------------------------------------------------------
# set_page_config must be called before other Streamlit commands.

st.set_page_config(
    page_title="Agentic Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# CSS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CSS_PATH = BASE_DIR / "styles.css"


def load_css():
    """
    Load styles.css from the same directory as this Python file.
    """
    try:
        with open(
            CSS_PATH,
            "r",
            encoding="utf-8",
        ) as css_file:
            st.markdown(
                f"<style>{css_file.read()}</style>",
                unsafe_allow_html=True,
            )
    except FileNotFoundError:
        st.warning(
            f"CSS file was not found: {CSS_PATH}"
        )


load_css()

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------


def generate_thread_id():
    """
    Generate a unique internal identifier for each conversation.
    """
    return str(uuid.uuid4())


def cohere_text_only(content):
    """
    Convert Cohere message content into plain text.
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


def message_content_to_text(content):
    """
    Convert message content to displayable text for either provider.
    """
    if isinstance(content, str):
        return cohere_text_only(content)

    return cohere_text_only(content)


def clean_title(title):
    """
    Clean a generated title before displaying it.
    """
    if not title:
        return "New Conversation"

    title = message_content_to_text(title)

    # Remove leading and trailing whitespace.
    title = title.strip()

    # Remove quotation marks.
    title = title.strip('"').strip("'")

    # Remove accidental Markdown heading characters.
    title = title.lstrip("#").strip()

    # Keep only the first line.
    lines = title.splitlines()

    if lines:
        title = lines[0].strip()

    # Keep the final title within 45 characters.
    if len(title) > 45:
        title = title[:42].rstrip() + "..."

    return title or "New Conversation"


def generate_chat_title(user_message):
    """
    Ask the selected LLM to generate a short conversation title.

    This request is sent directly to the LLM so the title prompt
    is not stored in the LangGraph conversation history.
    """
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

    response = llm.invoke(prompt)

    title = message_content_to_text(
        response.content
    )

    return clean_title(title)


def create_fallback_title(user_message):
    """
    Create a title from the first six words when the title LLM
    request fails or when an older conversation has no saved title.
    """
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

    return clean_title(fallback_title)


def add_thread(thread_id):
    """
    Add a thread ID to the sidebar list without duplicates.
    """
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(
            thread_id
        )


def reset_chat():
    """
    Create a new empty conversation.
    """
    new_thread_id = generate_thread_id()

    st.session_state["thread_id"] = new_thread_id
    st.session_state["message_history"] = []

    add_thread(new_thread_id)

    # Do not save this temporary title to SQLite yet.
    # Save the title only after the first user message.
    st.session_state["thread_titles"][
        new_thread_id
    ] = "New Conversation"


def delete_conversation(thread_id):
    """
    Delete a conversation from SQLite and Streamlit session state.
    """
    try:
        # Delete checkpoints, writes, and title from SQLite.
        delete_thread(thread_id)

        # Remove the thread from the sidebar list.
        if thread_id in st.session_state["chat_threads"]:
            st.session_state["chat_threads"].remove(
                thread_id
            )

        # Remove its title from session state.
        st.session_state["thread_titles"].pop(
            thread_id,
            None,
        )

        # If the currently selected conversation was deleted,
        # create a fresh empty conversation.
        if thread_id == st.session_state["thread_id"]:
            new_thread_id = generate_thread_id()

            st.session_state["thread_id"] = new_thread_id
            st.session_state["message_history"] = []

            st.session_state["chat_threads"].append(
                new_thread_id
            )

            st.session_state["thread_titles"][
                new_thread_id
            ] = "New Conversation"

        return True

    except Exception as error:
        print(
            f"Failed to delete conversation {thread_id}:",
            error,
        )

        st.error(
            "The conversation could not be deleted."
        )

        return False


def load_conversation(thread_id):
    """
    Load one conversation from LangGraph using its thread ID.
    """
    try:
        state = chatbot.get_state(
            config={
                "configurable": {
                    "thread_id": thread_id
                }
            }
        )

        if not state or not state.values:
            return []

        return state.values.get(
            "messages",
            [],
        )

    except Exception as error:
        print(
            f"Failed to load conversation {thread_id}:",
            error,
        )

        return []


def convert_saved_messages(messages):
    """
    Convert LangChain messages into dictionaries that Streamlit
    can display.
    """
    converted_messages = []

    for message in messages:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            # Ignore SystemMessage, ToolMessage, and other types.
            continue

        content = message_content_to_text(
            message.content
        )

        converted_messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    return converted_messages


def create_title_from_saved_conversation(thread_id):
    """
    Create and save a fallback title for an older conversation
    that does not yet have a persisted title.
    """
    saved_messages = load_conversation(
        thread_id
    )

    for message in saved_messages:
        if isinstance(message, HumanMessage):
            first_user_message = message_content_to_text(
                message.content
            )

            title = create_fallback_title(
                first_user_message
            )

            if title != "New Conversation":
                try:
                    save_thread_title(
                        thread_id,
                        title,
                    )
                except Exception as error:
                    print(
                        "Failed to save backfilled title:",
                        error,
                    )

            return title

    return "New Conversation"


# ------------------------------------------------------------
# Page heading
# ------------------------------------------------------------

st.markdown(
    """
    <div class="main-title">
        🤖 Agentic AI Assistant
    </div>

    <div class="subtitle">
        LangGraph • Cohere/OpenAI • Multi-Thread Memory
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Initialize Streamlit session state
# ------------------------------------------------------------

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if (
    "chat_threads" not in st.session_state
    or "thread_titles" not in st.session_state
):
    saved_conversations = (
        get_all_threads_with_titles()
    )

    # Create fallback titles for conversations that existed before
    # title persistence was added.
    for saved_thread_id, saved_title in list(
        saved_conversations.items()
    ):
        if (
            not saved_title
            or saved_title == "New Conversation"
        ):
            saved_conversations[saved_thread_id] = (
                create_title_from_saved_conversation(
                    saved_thread_id
                )
            )

    st.session_state["chat_threads"] = list(
        saved_conversations.keys()
    )

    st.session_state["thread_titles"] = dict(
        saved_conversations
    )

# Ensure the currently selected thread is shown in the sidebar.
add_thread(
    st.session_state["thread_id"]
)

# Give a newly created empty thread a temporary title.
current_thread_id = st.session_state["thread_id"]

if (
    current_thread_id
    not in st.session_state["thread_titles"]
):
    st.session_state["thread_titles"][
        current_thread_id
    ] = "New Conversation"

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.title("My Conversations")

if st.sidebar.button(
    "New Chat",
    use_container_width=True,
):
    reset_chat()
    st.rerun()

# New conversations are appended to the list.
# Reversing displays the newest conversation first.
for thread_id in reversed(
    st.session_state["chat_threads"]
):
    conversation_title = (
        st.session_state["thread_titles"].get(
            thread_id,
            "New Conversation",
        )
    )

    if thread_id == st.session_state["thread_id"]:
        button_label = f"💬 {conversation_title}"
    else:
        button_label = conversation_title

    # Create two columns:
    # one for the conversation and one for delete.
    conversation_column, delete_column = (
        st.sidebar.columns([5, 1])
    )

    with conversation_column:
        if st.button(
            button_label,
            key=f"conversation_{thread_id}",
            use_container_width=True,
        ):
            st.session_state["thread_id"] = thread_id

            saved_messages = load_conversation(
                thread_id
            )

            st.session_state["message_history"] = (
                convert_saved_messages(
                    saved_messages
                )
            )

            st.rerun()

    with delete_column:
        if st.button(
            "🗑️",
            key=f"delete_{thread_id}",
            help=f"Delete {conversation_title}",
            use_container_width=True,
        ):
            deleted = delete_conversation(
                thread_id
            )

            if deleted:
                st.rerun()

# ------------------------------------------------------------
# Display current conversation
# ------------------------------------------------------------

for message in st.session_state["message_history"]:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

# ------------------------------------------------------------
# Chat input
# ------------------------------------------------------------

user_input = st.chat_input(
    "Type your message here"
)

# ------------------------------------------------------------
# Process a new user message
# ------------------------------------------------------------

if user_input:
    current_thread_id = (
        st.session_state["thread_id"]
    )

    existing_title = (
        st.session_state["thread_titles"].get(
            current_thread_id,
            "New Conversation",
        )
    )

    should_generate_title = (
        not existing_title
        or existing_title == "New Conversation"
    )

    generated_title = None

    # Generate the title only for the first user message.
    if should_generate_title:
        try:
            generated_title = generate_chat_title(
                user_input
            )

        except Exception as title_error:
            generated_title = create_fallback_title(
                user_input
            )

            print(
                "Conversation title generation failed:",
                title_error,
            )

        st.session_state["thread_titles"][
            current_thread_id
        ] = generated_title

    # Store the user message in Streamlit session state.
    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    # Display the user message.
    with st.chat_message("user"):
        st.markdown(user_input)

    config = {
        "configurable": {
            "thread_id": current_thread_id
        }
    }

    try:
        # Stream and display the assistant response.
        with st.chat_message("assistant"):
            ai_message = st.write_stream(
                (
                    message_content_to_text(
                        message_chunk.content
                    )
                    for message_chunk, metadata
                    in chatbot.stream(
                        {
                            "messages": [
                                HumanMessage(
                                    content=user_input
                                )
                            ]
                        },
                        config=config,
                        stream_mode="messages",
                    )
                    if isinstance(
                        message_chunk,
                        AIMessage,
                    )
                )
            )

        # Store the complete assistant response in the UI session.
        st.session_state["message_history"].append(
            {
                "role": "assistant",
                "content": ai_message,
            }
        )

        # Save the generated title only after LangGraph successfully
        # creates the conversation checkpoint.
        if generated_title:
            try:
                save_thread_title(
                    current_thread_id,
                    generated_title,
                )

            except Exception as database_error:
                print(
                    "Failed to save conversation title:",
                    database_error,
                )

        st.rerun()

    except Exception as chatbot_error:
        st.error(
            "The assistant could not generate a response. "
            "Please try again."
        )

        print(
            "Chatbot processing failed:",
            chatbot_error,
        )

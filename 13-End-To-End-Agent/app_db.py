import hashlib
import tempfile
import uuid
from pathlib import Path

import streamlit as st

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)


# ------------------------------------------------------------
# Backend imports
# ------------------------------------------------------------

from backend.conversations import (
    delete_thread,
    get_all_threads_with_titles,
    save_thread_title,
)

from backend.graph import chatbot

from backend.message_utils import (
    message_content_to_text,
)

from tools.rag.ingest import ingest_pdf

from backend.titles import (
    create_fallback_title,
    generate_chat_title,
)


# ------------------------------------------------------------
# Streamlit configuration
# ------------------------------------------------------------

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
    Generate a unique internal identifier
    for each conversation.
    """

    return str(uuid.uuid4())


def add_thread(thread_id):
    """
    Add a thread ID to the sidebar list
    without duplicates.
    """

    if thread_id not in st.session_state["chat_threads"]:

        st.session_state[
            "chat_threads"
        ].append(
            thread_id
        )


def reset_chat():
    """
    Create a new conversation.

    If the current conversation is already empty,
    reuse it instead of creating another empty
    "New Conversation".
    """

    # If current chat is already empty,
    # just keep using it.
    if not st.session_state["message_history"]:

        current_thread_id = (
            st.session_state["thread_id"]
        )

        st.session_state["thread_titles"][
            current_thread_id
        ] = "New Conversation"

        return

    # Otherwise create a genuinely new conversation.
    new_thread_id = generate_thread_id()

    st.session_state["thread_id"] = new_thread_id

    st.session_state["message_history"] = []

    add_thread(new_thread_id)

    st.session_state["thread_titles"][
        new_thread_id
    ] = "New Conversation"
    """
    Create a new empty conversation.
    """

    new_thread_id = generate_thread_id()

    st.session_state[
        "thread_id"
    ] = new_thread_id

    st.session_state[
        "message_history"
    ] = []

    add_thread(
        new_thread_id
    )

    # Do not save this temporary title
    # to SQLite yet.
    #
    # Save the title only after the
    # first user message.

    st.session_state[
        "thread_titles"
    ][
        new_thread_id
    ] = "New Conversation"


def delete_conversation(thread_id):
    """
    Delete a conversation from SQLite
    and Streamlit session state.
    """

    try:

        # ----------------------------------------------------
        # Delete checkpoints and title from SQLite
        # ----------------------------------------------------

        delete_thread(
            thread_id
        )

        # ----------------------------------------------------
        # Remove thread from sidebar list
        # ----------------------------------------------------

        if (
            thread_id
            in st.session_state[
                "chat_threads"
            ]
        ):

            st.session_state[
                "chat_threads"
            ].remove(
                thread_id
            )

        # ----------------------------------------------------
        # Remove title from session state
        # ----------------------------------------------------

        st.session_state[
            "thread_titles"
        ].pop(
            thread_id,
            None,
        )

        # ----------------------------------------------------
        # If currently selected conversation
        # was deleted, create a new conversation.
        # ----------------------------------------------------

        if (
            thread_id
            == st.session_state[
                "thread_id"
            ]
        ):

            new_thread_id = (
                generate_thread_id()
            )

            st.session_state[
                "thread_id"
            ] = new_thread_id

            st.session_state[
                "message_history"
            ] = []

            st.session_state[
                "chat_threads"
            ].append(
                new_thread_id
            )

            st.session_state[
                "thread_titles"
            ][
                new_thread_id
            ] = "New Conversation"

        return True

    except Exception as error:

        print(
            f"Failed to delete conversation "
            f"{thread_id}:",
            error,
        )

        st.error(
            "The conversation could not be deleted."
        )

        return False


def load_conversation(thread_id):
    """
    Load one conversation from LangGraph
    using its thread ID.
    """

    try:

        state = chatbot.get_state(
            config={
                "configurable": {
                    "thread_id": thread_id
                }
            }
        )

        if (
            not state
            or not state.values
        ):

            return []

        return state.values.get(
            "messages",
            [],
        )

    except Exception as error:

        print(
            f"Failed to load conversation "
            f"{thread_id}:",
            error,
        )

        return []


def convert_saved_messages(messages):
    """
    Convert LangChain messages into dictionaries
    that Streamlit can display.

    ToolMessage is preserved so tool usage remains
    visible after reruns and when loading a conversation.
    """

    converted_messages = []

    for message in messages:

        if isinstance(
            message,
            HumanMessage,
        ):

            converted_messages.append(
                {
                    "role": "user",
                    "content": message_content_to_text(
                        message.content
                    ),
                }
            )

        elif isinstance(
            message,
            AIMessage,
        ):

            content = message_content_to_text(
                message.content
            )

            # Do not display the AI tool-call message itself.
            # The corresponding ToolMessage below is displayed
            # as the compact "Tool used" entry.
            if content:
                converted_messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )

        elif isinstance(
            message,
            ToolMessage,
        ):

            converted_messages.append(
                {
                    "role": "tool",
                    "tool_name": (
                        message.name
                        or "Tool"
                    ),
                    "content": message_content_to_text(
                        message.content
                    ),
                }
            )

        # SystemMessage and other message types are ignored.

    return converted_messages


def create_title_from_saved_conversation(
    thread_id,
):
    """
    Create and save a fallback title for
    an older conversation that does not
    yet have a persisted title.
    """

    saved_messages = load_conversation(
        thread_id
    )

    for message in saved_messages:

        if isinstance(
            message,
            HumanMessage,
        ):

            first_user_message = (
                message_content_to_text(
                    message.content
                )
            )

            title = create_fallback_title(
                first_user_message
            )

            if (
                title
                != "New Conversation"
            ):

                try:

                    save_thread_title(
                        thread_id,
                        title,
                    )

                except Exception as error:

                    print(
                        "Failed to save "
                        "backfilled title:",
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
        LangGraph • OpenAI • Multi-Thread Memory
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Initialize Streamlit session state
# ------------------------------------------------------------

if (
    "message_history"
    not in st.session_state
):

    st.session_state[
        "message_history"
    ] = []


if "indexed_files" not in st.session_state:

    st.session_state["indexed_files"] = {}

if (
    "thread_id"
    not in st.session_state
):

    st.session_state[
        "thread_id"
    ] = generate_thread_id()


# ------------------------------------------------------------
# Load saved conversations
# ------------------------------------------------------------

if (
    "chat_threads"
    not in st.session_state
    or
    "thread_titles"
    not in st.session_state
):

    saved_conversations = (
        get_all_threads_with_titles()
    )

    # --------------------------------------------------------
    # Create fallback titles for conversations
    # that existed before title persistence
    # was added.
    # --------------------------------------------------------

    for (
        saved_thread_id,
        saved_title,
    ) in list(
        saved_conversations.items()
    ):

        if (
            not saved_title
            or saved_title
            == "New Conversation"
        ):

            saved_conversations[
                saved_thread_id
            ] = (
                create_title_from_saved_conversation(
                    saved_thread_id
                )
            )

    st.session_state[
        "chat_threads"
    ] = list(
        saved_conversations.keys()
    )

    st.session_state[
        "thread_titles"
    ] = dict(
        saved_conversations
    )


# ------------------------------------------------------------
# Ensure currently selected thread
# is shown in sidebar.
# ------------------------------------------------------------

add_thread(
    st.session_state[
        "thread_id"
    ]
)


# ------------------------------------------------------------
# Give newly created empty thread
# a temporary title.
# ------------------------------------------------------------

current_thread_id = (
    st.session_state[
        "thread_id"
    ]
)


if (
    current_thread_id
    not in st.session_state[
        "thread_titles"
    ]
):

    st.session_state[
        "thread_titles"
    ][
        current_thread_id
    ] = "New Conversation"


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.title(
    "My Conversations"
)


# ------------------------------------------------------------
# New Chat button
# ------------------------------------------------------------

if st.sidebar.button(
    "New Chat",
    use_container_width=True,
):

    reset_chat()

    st.rerun()


# ------------------------------------------------------------
# Conversation list
# ------------------------------------------------------------

# New conversations are appended to the list.
# Reversing displays the newest conversation first.

for thread_id in reversed(
    st.session_state[
        "chat_threads"
    ]
):

    conversation_title = (
        st.session_state[
            "thread_titles"
        ].get(
            thread_id,
            "New Conversation",
        )
    )

    # --------------------------------------------------------
    # Highlight currently selected conversation
    # --------------------------------------------------------

    if (
        thread_id
        == st.session_state[
            "thread_id"
        ]
    ):

        button_label = (
            f"💬 {conversation_title}"
        )

    else:

        button_label = conversation_title

    # --------------------------------------------------------
    # Create two columns:
    # one for conversation
    # one for delete
    # --------------------------------------------------------

    (
        conversation_column,
        delete_column,
    ) = st.sidebar.columns(
        [5, 1]
    )

    # --------------------------------------------------------
    # Conversation button
    # --------------------------------------------------------

    with conversation_column:

        if st.button(
            button_label,
            key=f"conversation_{thread_id}",
            use_container_width=True,
        ):

            st.session_state[
                "thread_id"
            ] = thread_id

            saved_messages = (
                load_conversation(
                    thread_id
                )
            )

            st.session_state[
                "message_history"
            ] = (
                convert_saved_messages(
                    saved_messages
                )
            )

            st.rerun()

    # --------------------------------------------------------
    # Delete button
    # --------------------------------------------------------

    with delete_column:

        if st.button(
            "🗑️",
            key=f"delete_{thread_id}",
            help=f"Delete {conversation_title}",
            use_container_width=True,
        ):

            deleted = (
                delete_conversation(
                    thread_id
                )
            )

            if deleted:

                st.rerun()


# ------------------------------------------------------------
# Display current conversation
# ------------------------------------------------------------

for message in (
    st.session_state[
        "message_history"
    ]
):

    if message["role"] == "tool":

        # Compact tool display, matching the original UI.
        with st.status(
            f"🔧 Tool used: {message.get('tool_name', 'Tool')}",
            state="complete",
        ):

            if message.get("content"):
                st.code(
                    message["content"]
                )

    else:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


# ------------------------------------------------------------
# Compact uploaded documents
# ------------------------------------------------------------

if st.session_state["indexed_files"]:

    with st.expander(
        f"📚 Uploaded Documents ({len(st.session_state['indexed_files'])})",
        expanded=False,
    ):

        for file_info in st.session_state["indexed_files"].values():

            st.markdown(
                f"📄 **{file_info['name']}**"
            )

            st.caption(
                f"✅ Ready • "
                f"{file_info['pages']} pages • "
                f"{file_info['chunks']} chunks"
            )


# ------------------------------------------------------------
# Chat input with PDF upload
# ------------------------------------------------------------

chat_input = st.chat_input(
    "Ask anything...",
    accept_file="multiple",
    file_type=["pdf"],
)


# ------------------------------------------------------------
# Process uploaded PDFs and new user message
# ------------------------------------------------------------

if chat_input:

    # --------------------------------------------------------
    # Streamlit returns a ChatInputValue when file upload
    # is enabled. Extract the text prompt and files safely.
    # --------------------------------------------------------

    user_input = chat_input.text

    uploaded_files = chat_input.files

    # --------------------------------------------------------
    # Index newly uploaded PDF files
    # --------------------------------------------------------

    newly_indexed_files = []

    if uploaded_files:

        for uploaded_file in uploaded_files:

            file_bytes = uploaded_file.getvalue()

            file_hash = hashlib.sha256(
                file_bytes
            ).hexdigest()

            # Avoid indexing the same file more than once
            # during the current Streamlit session.
            if file_hash in st.session_state["indexed_files"]:
                continue

            with st.status(
                f"📄 Processing {uploaded_file.name}...",
                expanded=True,
            ) as upload_status:

                try:

                    suffix = Path(
                        uploaded_file.name
                    ).suffix or ".pdf"

                    temp_file_path = None

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix,
                    ) as temp_file:

                        temp_file.write(
                            file_bytes
                        )

                        temp_file_path = Path(
                            temp_file.name
                        )

                    try:

                        ingest_result = ingest_pdf(
                            temp_file_path
                        )

                    finally:

                        if (
                            temp_file_path
                            and temp_file_path.exists()
                        ):

                            temp_file_path.unlink()

                    file_info = {
                        "name": uploaded_file.name,
                        "pages": ingest_result["pages"],
                        "chunks": ingest_result["chunks"],
                        "hash": file_hash,
                    }

                    # ------------------------------------------------
                    # IMPORTANT:
                    # Save the completed upload in session state.
                    # This survives st.rerun() and is displayed above
                    # the chat input on every run.
                    # ------------------------------------------------

                    st.session_state[
                        "indexed_files"
                    ][
                        file_hash
                    ] = file_info

                    newly_indexed_files.append(
                        file_info
                    )

                    upload_status.update(
                        label=(
                            f"✅ {uploaded_file.name} "
                            "is ready"
                        ),
                        state="complete",
                        expanded=False,
                    )

                except Exception as upload_error:

                    upload_status.update(
                        label=(
                            f"❌ Failed to process "
                            f"{uploaded_file.name}"
                        ),
                        state="error",
                        expanded=True,
                    )

                    st.error(
                        f"Could not process "
                        f"{uploaded_file.name}: "
                        f"{upload_error}"
                    )

    # --------------------------------------------------------
    # If the user uploaded a file without entering a question,
    # do not send an empty message to LangGraph.
    # --------------------------------------------------------

    if not user_input.strip():

        if newly_indexed_files:

            st.success(
                "📚 Your document is indexed and ready "
                "for questions."
            )

        st.rerun()

    # --------------------------------------------------------
    # Current thread
    # --------------------------------------------------------

    current_thread_id = (
        st.session_state[
            "thread_id"
        ]
    )

    # --------------------------------------------------------
    # Check existing title
    # --------------------------------------------------------

    existing_title = (
        st.session_state[
            "thread_titles"
        ].get(
            current_thread_id,
            "New Conversation",
        )
    )

    should_generate_title = (
        not existing_title
        or
        existing_title
        == "New Conversation"
    )

    generated_title = None

    # --------------------------------------------------------
    # Generate title only for first
    # user message
    # --------------------------------------------------------

    if should_generate_title:

        try:

            generated_title = (
                generate_chat_title(
                    user_input
                )
            )

        except Exception as title_error:

            generated_title = (
                create_fallback_title(
                    user_input
                )
            )

            print(
                "Conversation title "
                "generation failed:",
                title_error,
            )

        st.session_state[
            "thread_titles"
        ][
            current_thread_id
        ] = generated_title

    # --------------------------------------------------------
    # Store user message in Streamlit
    # session state
    # --------------------------------------------------------

    st.session_state[
        "message_history"
    ].append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_input
        )

    # --------------------------------------------------------
    # LangGraph thread configuration
    # --------------------------------------------------------

    config = {
        "configurable": {
            "thread_id": current_thread_id
        }
    }

    # --------------------------------------------------------
    # Stream assistant response
    # --------------------------------------------------------

    try:

        with st.chat_message(
            "assistant"
        ):

            # ------------------------------------------------
            # Placeholder for final AI response
            # ------------------------------------------------

            response_placeholder = st.empty()

            full_response = ""

            # ------------------------------------------------
            # Stream LangGraph messages
            # ------------------------------------------------

            for (
                message_chunk,
                metadata,
            ) in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(
                            content=user_input
                        )
                    ]
                },
                config=config,
                stream_mode="messages",
            ):

                # ============================================
                # TOOL MESSAGE
                # ============================================

                if isinstance(
                    message_chunk,
                    ToolMessage,
                ):

                    tool_name = (
                        message_chunk.name
                        or "Tool"
                    )

                    tool_result = (
                        message_content_to_text(
                            message_chunk.content
                        )
                    )

                    # ----------------------------------------
                    # Display tool execution
                    # ----------------------------------------

                    # Keep the original compact tool UI.
                    with st.status(
                        f"🔧 Tool used: {tool_name}",
                        state="complete",
                    ):

                        if tool_result:
                            st.code(
                                tool_result
                            )

                    # Persist the tool message in the same conversation
                    # history so it is restored after st.rerun().
                    st.session_state[
                        "message_history"
                    ].append(
                        {
                            "role": "tool",
                            "tool_name": tool_name,
                            "content": tool_result,
                        }
                    )

                # ============================================
                # AI MESSAGE
                # ============================================

                elif isinstance(
                    message_chunk,
                    AIMessage,
                ):

                    content = (
                        message_content_to_text(
                            message_chunk.content
                        )
                    )

                    if content:

                        full_response += content

                        response_placeholder.markdown(
                            full_response
                        )

            # ------------------------------------------------
            # Store complete assistant response
            # ------------------------------------------------

            st.session_state[
                "message_history"
            ].append(
                {
                    "role": "assistant",
                    "content": full_response,
                }
            )

        # ----------------------------------------------------
        # Save generated title only after
        # LangGraph successfully creates
        # the conversation checkpoint.
        # ----------------------------------------------------

        if generated_title:

            try:

                save_thread_title(
                    current_thread_id,
                    generated_title,
                )

            except Exception as database_error:

                print(
                    "Failed to save "
                    "conversation title:",
                    database_error,
                )

        # ----------------------------------------------------
        # Refresh UI
        # ----------------------------------------------------

        st.rerun()

    except Exception as chatbot_error:

        st.error(
            "The assistant could not generate "
            "a response. Please try again."
        )

        print(
            "Chatbot processing failed:",
            chatbot_error,
        )

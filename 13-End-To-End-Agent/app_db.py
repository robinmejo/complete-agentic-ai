import uuid
from pathlib import Path

import streamlit as st

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
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

from backend.llm import llm

from backend.message_utils import (
    message_content_to_text,
)

from backend.titles import (
    create_fallback_title,
    generate_chat_title,
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
    Generate a unique internal identifier
    for each conversation.
    """

    return str(uuid.uuid4())


def add_thread(thread_id):
    """
    Add a thread ID to the sidebar list
    without duplicates.
    """

    if (
        thread_id
        not in st.session_state["chat_threads"]
    ):

        st.session_state[
            "chat_threads"
        ].append(
            thread_id
        )


def reset_chat():
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
    """

    converted_messages = []

    for message in messages:

        if isinstance(
            message,
            HumanMessage,
        ):

            role = "user"

        elif isinstance(
            message,
            AIMessage,
        ):

            role = "assistant"

        else:

            # Ignore SystemMessage,
            # ToolMessage and other types.

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
        LangGraph • Cohere/OpenAI • Multi-Thread Memory
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

            ai_message = st.write_stream(
                (
                    message_content_to_text(
                        message_chunk.content
                    )

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
                    )

                    if isinstance(
                        message_chunk,
                        AIMessage,
                    )
                )
            )


        # ----------------------------------------------------
        # Store complete assistant response
        # in Streamlit session state
        # ----------------------------------------------------

        st.session_state[
            "message_history"
        ].append(
            {
                "role": "assistant",
                "content": ai_message,
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
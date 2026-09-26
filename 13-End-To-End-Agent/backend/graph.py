from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from .config import USE_OPENAI
from .database import checkpoint
from .llm import llm
from .message_utils import cohere_text_only
from .state import ChatState


# ------------------------------------------------------------
# LangGraph chatbot node
# ------------------------------------------------------------

def chat_node(state: ChatState):

    messages = state["messages"]

    # --------------------------------------------------------
    # Normalize Cohere message content
    # --------------------------------------------------------

    if not USE_OPENAI:

        normalized_messages = []

        for message in messages:

            if isinstance(
                message,
                HumanMessage,
            ):

                normalized_messages.append(
                    HumanMessage(
                        content=cohere_text_only(
                            message.content
                        )
                    )
                )

            elif isinstance(
                message,
                AIMessage,
            ):

                normalized_messages.append(
                    AIMessage(
                        content=cohere_text_only(
                            message.content
                        )
                    )
                )

            else:

                normalized_messages.append(
                    message
                )

        messages = normalized_messages

    # --------------------------------------------------------
    # Invoke LLM
    # --------------------------------------------------------

    response = llm.invoke(
        messages
    )

    return {
        "messages": [
            response
        ]
    }


# ------------------------------------------------------------
# Build LangGraph
# ------------------------------------------------------------

def build_graph():

    graph = StateGraph(
        ChatState
    )

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

    return graph.compile(
        checkpointer=checkpoint
    )


# ------------------------------------------------------------
# Compiled chatbot
# ------------------------------------------------------------

chatbot = build_graph()
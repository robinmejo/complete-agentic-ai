from langchain_core.messages import (
    AIMessage,
    SystemMessage,
)

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from langgraph.prebuilt import ToolNode

from .database import checkpoint
from .llm import llm
from .state import ChatState

from tools.calculator import calculator
from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from langgraph.prebuilt import ToolNode

from .database import checkpoint
from .llm import llm
from .state import ChatState

from tools.calculator import calculator
from tools.tavily_search import tavily_search
from tools.stock_price import get_stock_price
from tools.weather import get_weather
from tools.rag.search import document_search

# ============================================================
# TOOLS
# ============================================================

tools = [
    calculator,
    tavily_search,
    get_stock_price,
    get_weather,
    document_search,
]


# ============================================================
# LLM WITH TOOLS
# ============================================================

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# TOOL NODE
# ============================================================

tool_node = ToolNode(tools)


# ============================================================
# CHAT NODE
# ============================================================

def chat_node(state: ChatState):

    messages = state["messages"]

    # --------------------------------------------------------
    # System instruction for mathematical formatting
    # --------------------------------------------------------

    system_message = SystemMessage(
        content=(
            "You are an AI assistant with access to uploaded documents.\n\n"

            "IMPORTANT RULES:\n"
            "1. When a question could be related to the uploaded documents, "
            "ALWAYS use the document_search tool first.\n"

            "2. If the answer is found in the uploaded documents, "
            "answer using the retrieved document information.\n"

            "3. If the answer is NOT found in the uploaded documents, "
            "clearly mention that the information was not found in the "
            "uploaded documents, and then answer using your general "
            "knowledge when possible.\n"

            "4. Clearly distinguish between information from the uploaded "
            "documents and information from your general knowledge.\n"

            "5. Do not invent or claim that information came from the "
            "uploaded documents when it did not.\n\n"

            "When displaying mathematical calculations, "
            "use plain text notation. "
            "Do not use LaTeX commands such as \\times, "
            "\\frac, or math delimiters. "
            "Use normal symbols such as x, /, and =."
        )
    )

    messages = [
        system_message,
        *messages,
    ]

    # --------------------------------------------------------
    # Call LLM with tools
    # --------------------------------------------------------

    response = llm_with_tools.invoke(
        messages
    )

    return {
        "messages": [
            response
        ]
    }

# ============================================================
# ROUTING FUNCTION
# ============================================================

def should_continue(state: ChatState):
    """
    Decide whether the graph should execute tools
    or finish the conversation.
    """

    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage):

        if last_message.tool_calls:
            return "tools"

    return END


# ============================================================
# BUILD GRAPH
# ============================================================

def build_graph():

    graph = StateGraph(ChatState)

    # --------------------------------------------------------
    # Add nodes
    # --------------------------------------------------------

    graph.add_node(
        "chat_node",
        chat_node,
    )

    graph.add_node(
        "tools",
        tool_node,
    )

    # --------------------------------------------------------
    # START → chat_node
    # --------------------------------------------------------

    graph.add_edge(
        START,
        "chat_node",
    )

    # --------------------------------------------------------
    # chat_node → tools OR END
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "chat_node",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )

    # --------------------------------------------------------
    # tools → chat_node
    # --------------------------------------------------------

    graph.add_edge(
        "tools",
        "chat_node",
    )

    # --------------------------------------------------------
    # Compile graph with checkpointing
    # --------------------------------------------------------

    return graph.compile(
        checkpointer=checkpoint
    )


# ============================================================
# COMPILED CHATBOT
# ============================================================

chatbot = build_graph()
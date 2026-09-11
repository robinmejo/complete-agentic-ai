import os
import certifi
import requests
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_react_agent


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Agent",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main page */
    .main {
        padding-top: 2rem;
    }

    /* Tool cards */
    .tool-card {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-bottom: 10px;
        text-align: center;
    }

    .tool-icon {
        font-size: 25px;
    }

    .tool-name {
        font-weight: 600;
        margin-top: 5px;
    }

    .tool-description {
        font-size: 13px;
        opacity: 0.65;
    }

    /* Response */
    .response-title {
        font-size: 20px;
        font-weight: 600;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    /* Footer */
    .footer {
        text-align: center;
        opacity: 0.5;
        font-size: 13px;
        margin-top: 40px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🤖 AI Agent")

    st.markdown(
        """
        This agent can **reason, act, and use tools** to answer your questions.
        """
    )

    st.divider()

    st.markdown("### 🛠️ Available Tools")

    st.markdown(
        """
        <div class="tool-card">
            <div class="tool-icon">🌐</div>
            <div class="tool-name">Web Search</div>
            <div class="tool-description">
                Search the web for current information
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="tool-card">
            <div class="tool-icon">🌤️</div>
            <div class="tool-name">Weather</div>
            <div class="tool-description">
                Get current weather information
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.caption("Powered by LangChain + OpenAI + Tavily")


# =========================================================
# MAIN HEADER
# =========================================================

# =========================================================
# MAIN HEADER
# =========================================================

st.title("🤖 Single AI Agent")

st.caption("Reason • Act • Use Tools • Answer")

# =========================================================
# DESCRIPTION
# =========================================================

st.info(
    "💡 Ask a question and the agent will decide which tools "
    "it needs to use to find the answer."
)


# =========================================================
# SEARCH TOOL
# =========================================================

search_tool = TavilySearchResults(
    tavily_api_key=TAVILY_API_KEY,
    search_engine="google",
    max_results=2,
    return_direct_answer=True,
)


# =========================================================
# WEATHER TOOL
# =========================================================

@tool
def get_weather_data(city: str) -> str:
    """
    Get current weather data for a given city.
    """

    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

    response = requests.get(url)
    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}"

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Description: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )


# =========================================================
# LLM
# =========================================================

llm = ChatOpenAI(
    model_name="gpt-4o",
    openai_api_key=OPENAI_API_KEY,
    temperature=0.0,
    max_tokens=1000,
)


# =========================================================
# PROMPT
# =========================================================

prompt = hub.pull(
    "hwchase17/react",
)


# =========================================================
# TOOLS
# =========================================================

tools = [
    search_tool,
    get_weather_data,
]


# =========================================================
# CREATE AGENT
# =========================================================

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)


# =========================================================
# AGENT EXECUTOR
# =========================================================

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)


# =========================================================
# USER INPUT
# =========================================================

st.markdown("### 💬 Ask your Agent")

user_input = st.text_area(
    "Your question",
    placeholder=(
        "Try something like:\n\n"
        "Find the capital of India and tell me its current weather."
    ),
    height=120,
    label_visibility="collapsed",
)


# =========================================================
# RUN AGENT
# =========================================================

if st.button(
    "🚀 Run Agent",
    use_container_width=True,
):

    if not user_input.strip():

        st.warning("Please enter a question first.")

    else:

        with st.spinner("🤖 Agent is thinking and using tools..."):

            response = agent_executor.invoke(
                {
                    "input": user_input
                }
            )

        st.markdown(
            '<div class="response-title">✨ Agent Response</div>',
            unsafe_allow_html=True,
        )

        st.success(response["output"])


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Built with 🧠 LangChain • 🔎 Tavily • 🌤️ Weatherstack • ⚡ Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
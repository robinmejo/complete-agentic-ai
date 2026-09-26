from langchain_core.tools import tool
from tavily import TavilyClient

from backend.config import TAVILY_API_KEY


@tool
def tavily_search(query: str) -> str:
    """Search the web for current or up-to-date information."""

    if not TAVILY_API_KEY:
        raise ValueError(
            "TAVILY_API_KEY was not found in .env"
        )

    tavily_client = TavilyClient(
        api_key=TAVILY_API_KEY
    )

    response = tavily_client.search(
        query=query,
        max_results=5,
    )

    results = response.get("results", [])

    if not results:
        return "No search results found."

    formatted_results = []

    for result in results:
        title = result.get("title", "")
        content = result.get("content", "")
        url = result.get("url", "")

        formatted_results.append(
            f"Title: {title}\n"
            f"Content: {content}\n"
            f"URL: {url}"
        )

    return "\n\n".join(formatted_results)
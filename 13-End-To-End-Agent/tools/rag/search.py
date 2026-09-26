from langchain_core.tools import tool

from .vector_store import get_vector_store


@tool
def document_search(query: str) -> str:
    """
    Search the user's uploaded documents
    for information relevant to the query.
    """

    vector_store = get_vector_store()

    documents = vector_store.similarity_search(
        query,
        k=4,
    )

    if not documents:

        return (
            "No relevant information was found "
            "in the uploaded documents."
        )

    results = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown",
        )

        page = document.metadata.get(
            "page",
            None,
        )

        if page is not None:

            page_number = page + 1

            source_info = (
                f"{source}, page {page_number}"
            )

        else:

            source_info = source

        results.append(
            f"Source: {source_info}\n"
            f"Content:\n"
            f"{document.page_content}"
        )

    return "\n\n---\n\n".join(
        results
    )
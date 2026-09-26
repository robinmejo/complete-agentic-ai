from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path: str | Path):
    """
    Load a PDF and return LangChain documents.
    """

    file_path = Path(file_path)

    loader = PyPDFLoader(
        str(file_path)
    )

    documents = loader.load()

    return documents
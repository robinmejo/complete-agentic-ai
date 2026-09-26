from pathlib import Path
import uuid

from .document_loader import load_pdf
from .document_splitter import split_documents
from .vector_store import get_vector_store


def ingest_pdf(file_path: str | Path):

    file_path = Path(file_path)

    # ---------------------------------------------
    # Load document
    # ---------------------------------------------

    documents = load_pdf(
        file_path
    )

    # ---------------------------------------------
    # Split into chunks
    # ---------------------------------------------

    chunks = split_documents(
        documents
    )

    # ---------------------------------------------
    # Add useful metadata
    # ---------------------------------------------

    for chunk in chunks:

        chunk.metadata[
            "source"
        ] = file_path.name

        chunk.metadata[
            "document_id"
        ] = str(
            uuid.uuid4()
        )

    # ---------------------------------------------
    # Store in Chroma
    # ---------------------------------------------

    vector_store = get_vector_store()

    vector_store.add_documents(
        documents=chunks
    )

    return {
        "file_name": file_path.name,
        "pages": len(documents),
        "chunks": len(chunks),
    }
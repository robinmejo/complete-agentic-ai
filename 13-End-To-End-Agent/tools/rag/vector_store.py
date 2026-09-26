from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


BASE_DIR = Path(__file__).resolve().parents[2]

CHROMA_DIR = BASE_DIR / "chroma_db"


COLLECTION_NAME = "agent_documents"


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


def get_vector_store():

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(
            CHROMA_DIR
        ),
    )

    return vector_store
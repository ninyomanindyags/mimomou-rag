"""Text splitting / chunking dokumen."""
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.helpers import load_config


def chunk_documents(documents, chunk_size: int | None = None, chunk_overlap: int | None = None):
    config = load_config()
    chunk_size = chunk_size or config["chunking"]["chunk_size"]
    chunk_overlap = chunk_overlap or config["chunking"]["chunk_overlap"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)

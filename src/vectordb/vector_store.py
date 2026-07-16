"""Load / build / persist Chroma vector store, untuk mode Baseline maupun SCG."""
import shutil
from pathlib import Path

from langchain_chroma import Chroma

from src.embeddings.embedder import load_embedder
from src.utils.helpers import load_config


def get_db_path(mode: str) -> str:
    config = load_config()["paths"]
    return config["chroma_baseline"] if mode == "Baseline" else config["chroma_scg"]


def load_vector_store(mode: str) -> Chroma:
    """Load koleksi Chroma yang SUDAH ADA untuk `mode` ("Baseline" / "SCG")."""
    db_path = get_db_path(mode)
    return Chroma(persist_directory=db_path, embedding_function=load_embedder())


def reset_vector_store(mode: str):
    """Hapus folder Chroma lama untuk `mode`, kalau ada."""
    db_path = get_db_path(mode)
    if Path(db_path).exists():
        shutil.rmtree(db_path)


def persist_documents(documents, mode: str) -> Chroma:
    """Embed `documents` lalu simpan sebagai koleksi Chroma baru."""
    db_path = get_db_path(mode)
    return Chroma.from_documents(
        documents=documents,
        embedding=load_embedder(),
        persist_directory=db_path,
    )

"""Embedding model loader.

BUG FIX: sebelumnya embedding_model di-instantiate langsung sebagai
module-level global di retriever.py -- artinya model ke-load ke memori
setiap kali modul itu di-import, walau belum tentu dipakai, dan tanpa
guard singleton (beda dengan reranker.py yang sudah pakai pola singleton).
Di sini load_embedder() malas (lazy) dan cuma load sekali per proses.
"""
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.helpers import load_config

_embedding_model = None


def load_embedder():
    global _embedding_model
    if _embedding_model is None:
        config = load_config()
        _embedding_model = HuggingFaceEmbeddings(
            model_name=config["embedding"]["model_name"]
        )
    return _embedding_model

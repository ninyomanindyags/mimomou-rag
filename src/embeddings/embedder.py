"""Embedding model loader.

BUG FIX: sebelumnya embedding_model di-instantiate langsung sebagai
module-level global di retriever.py -- artinya model ke-load ke memori
setiap kali modul itu di-import, walau belum tentu dipakai, dan tanpa
guard singleton (beda dengan reranker.py yang sudah pakai pola singleton).
Di sini load_embedder() malas (lazy) dan cuma load sekali per proses.
"""
from langchain_huggingface import HuggingFaceEmbeddings # buat load model embedding dari HuggingFace

from src.utils.helpers import load_config

_embedding_model = None # global variable untuk menyimpan objek embedding model yang sudah di-load, supaya cuma load sekali per proses


def load_embedder():
    global _embedding_model # gunakan global variable _embedding_model
    if _embedding_model is None: # kalau belum buat objek embedding model, baru buat instance baru
        config = load_config()
        # buat load model embedding 
        _embedding_model = HuggingFaceEmbeddings(
            model_name=config["embedding"]["model_name"]
        )
    # mengembalikan objek embedding model yang sudah di-load (cuma load sekali per proses) 
    return _embedding_model

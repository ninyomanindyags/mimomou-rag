"""Build (atau rebuild) Chroma vector store untuk mode Baseline dari korpus PDF."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingestion.loader import load_pdfs
from src.chunking.chunker import chunk_documents
from src.vectordb.vector_store import reset_vector_store, persist_documents


def main():
    reset_vector_store("Baseline")

    print("Loading PDF...")
    documents = load_pdfs()
    print(f"Jumlah PDF : {len(documents)}")

    chunks = chunk_documents(documents)
    print(f"Jumlah Chunk : {len(chunks)}")

    print("Membuat ChromaDB Baseline...")
    db = persist_documents(chunks, "Baseline")

    print("Baseline selesai dibuat!")
    print("Jumlah dokumen :", db._collection.count())


if __name__ == "__main__":
    main()

"""Build (atau rebuild) Chroma vector store untuk mode Baseline dari korpus PDF."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document

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

    # tambahkan metadata agar konsisten dengan database SCG
    baseline_docs = []
    for i, chunk in enumerate(chunks):
        baseline_docs.append(
            Document(
                page_content=chunk.page_content,
                metadata={
                    **chunk.metadata,
                    "chunk_id": i,
                    "source_type": "original",
                },
            )
        )

    print("Membuat ChromaDB Baseline...")
    db = persist_documents(baseline_docs, "Baseline")

    print("Baseline selesai dibuat!")
    print("Jumlah dokumen :", db._collection.count())


if __name__ == "__main__":
    main()
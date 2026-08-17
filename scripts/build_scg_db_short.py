"""Build Chroma vector store untuk mode SCG_CONTEXTUAL_SHORT.

Sama seperti build_scg_db.py, tapi pakai SCG_PROMPT_SHORT (versi ringkas
ala Anthropic, 50-100 token) dan checkpoint terpisah -- supaya tidak
menimpa checkpoint/DB dari versi prompt yang lama (panjang/terstruktur).
"""
import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document

from src.ingestion.loader import load_pdfs
from src.chunking.chunker import chunk_documents
from src.llm.llm_client import load_scg_llm
from src.prompts.prompt_templates import SCG_PROMPT_SHORT
from src.vectordb.vector_store import reset_vector_store, persist_documents
from src.utils.helpers import load_checkpoint, append_checkpoint, load_config

MAX_RETRY = 3
DB_NAME = "SCG_CONTEXTUAL_SHORT"


def get_neighbor_context(chunks, index, max_chars=300):
    if index < 0 or index >= len(chunks):
        return ""
    return chunks[index].page_content[:max_chars]


def process_chunk(i, chunk, chunks, chat_model, checkpoint_path):
    source_current = chunk.metadata.get("source")

    prev_context = ""
    if i > 0 and chunks[i - 1].metadata.get("source") == source_current:
        prev_context = get_neighbor_context(chunks, i - 1)

    next_context = ""
    if i < len(chunks) - 1 and chunks[i + 1].metadata.get("source") == source_current:
        next_context = get_neighbor_context(chunks, i + 1)

    prompt = SCG_PROMPT_SHORT.invoke({
        "context": chunk.page_content,
        "prev_context": prev_context or "(tidak ada / chunk ini di awal dokumen)",
        "next_context": next_context or "(tidak ada / chunk ini di akhir dokumen)",
    })

    synthetic_context = None
    for retry in range(MAX_RETRY):
        try:
            response = chat_model.invoke(prompt)
            synthetic_context = response.content
            break
        except Exception as e:
            print(f"[Chunk {i}] Retry {retry + 1}/{MAX_RETRY} - {e}")
            time.sleep(5)

    if synthetic_context is None:
        print(f"[Chunk {i}] SKIP - gagal setelah {MAX_RETRY} retry")
        return None

    record = {
        "chunk_id": i,
        "page_content": chunk.page_content,
        "synthetic_context": synthetic_context,
        "metadata": chunk.metadata,
    }
    append_checkpoint(checkpoint_path, record)
    print(f"[Chunk {i}] selesai")
    return record


def main():
    config = load_config()
    checkpoint_path = config["paths"]["scg_checkpoint_short"]  # <- checkpoint BARU, beda file

    reset_vector_store(DB_NAME)

    print("Loading PDF...")
    documents = load_pdfs()
    print(f"Jumlah PDF   : {len(documents)}")

    chunks = chunk_documents(documents)
    print(f"Jumlah Chunk : {len(chunks)}")

    done = load_checkpoint(checkpoint_path, key_field="chunk_id")
    if done:
        print(f"Ditemukan checkpoint: {len(done)} chunk sudah selesai sebelumnya, akan di-skip.\n")

    pending = [(i, chunk) for i, chunk in enumerate(chunks) if i not in done]
    print(f"Sisa yang perlu diproses: {len(pending)}\n")

    if pending:
        chat_model = load_scg_llm()
        for i, chunk in pending:
            process_chunk(i, chunk, chunks, chat_model, checkpoint_path)

    done = load_checkpoint(checkpoint_path, key_field="chunk_id")
    print(f"\nTotal chunk berhasil diproses: {len(done)} / {len(chunks)}")

    if len(done) < len(chunks):
        missing = [i for i in range(len(chunks)) if i not in done]
        print(f"⚠️  {len(missing)} chunk gagal/skip: {missing}")
        print("Jalankan ulang script ini untuk retry chunk yang gagal (yang sudah sukses tidak akan diulang).")
        return

    synthetic_docs = []
    for i in sorted(done.keys()):
        record = done[i]

        contextualized_content = (
            f"{record['synthetic_context'].strip()}\n\n{record['page_content']}"
        )

        synthetic_docs.append(Document(
            page_content=contextualized_content,
            metadata={**record["metadata"], "chunk_id": i},
        ))

    print(f"\nTotal document (contextualized) : {len(synthetic_docs)}")
    print(f"\nMenyimpan ChromaDB {DB_NAME}...")
    db = persist_documents(synthetic_docs, DB_NAME)

    print(f"{DB_NAME} selesai dibuat!")
    print("Jumlah dokumen :", db._collection.count())


if __name__ == "__main__":
    main()
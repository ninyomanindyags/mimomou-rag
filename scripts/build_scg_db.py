"""Build (atau rebuild) Chroma vector store untuk mode SCG.

Generate synthetic context per chunk pakai LLM (dengan checkpoint biar
resumable kalau macet di tengah), lalu simpan pasangan original+synthetic
ke Chroma.
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
from src.prompts.prompt_templates import SCG_PROMPT
from src.vectordb.vector_store import reset_vector_store, persist_documents
from src.utils.helpers import load_checkpoint, append_checkpoint, load_config

MAX_RETRY = 3


def get_neighbor_context(chunks, index, max_chars=300):
    """
    Ambil potongan chunk tetangga (sebelum/sesudah) yang sudah ada di
    memori (BUKAN baca ulang PDF), dipotong ke max_chars biar prompt nggak
    kepanjangan -- cuma butuh "petunjuk topik", bukan isi lengkap.
    """
    if index < 0 or index >= len(chunks):
        return ""
    return chunks[index].page_content[:max_chars]


def process_chunk(i, chunk, chunks, chat_model, checkpoint_path):
    source_current = chunk.metadata.get("source")

    # Tetangga cuma dipakai kalau masih dari file PDF yang sama (biar nggak
    # nyambungin isi 2 dokumen yang beda topik cuma karena index-nya berurutan).
    prev_context = ""
    if i > 0 and chunks[i - 1].metadata.get("source") == source_current:
        prev_context = get_neighbor_context(chunks, i - 1)

    next_context = ""
    if i < len(chunks) - 1 and chunks[i + 1].metadata.get("source") == source_current:
        next_context = get_neighbor_context(chunks, i + 1)

    prompt = SCG_PROMPT.invoke({
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
    checkpoint_path = config["paths"]["scg_checkpoint"]

    # Checkpoint TIDAK dihapus di sini, biar bisa resume.
    reset_vector_store("SCG")

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
        # BUG FIX: sebelumnya script tetap lanjut membuat ChromaDB dari data
        # yang PARSIAL walau sudah tahu ada chunk yang gagal (cuma nge-print
        # warning, nggak berhenti). Ini beresiko besar untuk skripsi karena
        # DB SCG yang tidak lengkap akan bikin perbandingan Baseline vs SCG
        # jadi tidak adil/valid tanpa disadari. Sekarang PROSES DIHENTIKAN
        # sampai semua chunk berhasil, konsisten dengan pola yang sudah
        # dipakai di evaluate_retrieval.py (still_failed -> return).
        return

    synthetic_docs = []
    for i in sorted(done.keys()):
        record = done[i]
        synthetic_docs.append(Document(
            page_content=record["page_content"],
            metadata={**record["metadata"], "source_type": "original", "chunk_id": i},
        ))
        synthetic_docs.append(Document(
            page_content=record["synthetic_context"],
            metadata={**record["metadata"], "source_type": "synthetic", "chunk_id": i},
        ))

    print(f"\nTotal document (original + synthetic) : {len(synthetic_docs)}")
    print("\nMenyimpan ChromaDB SCG...")
    db = persist_documents(synthetic_docs, "SCG")

    print("SCG selesai dibuat!")
    print("Jumlah dokumen :", db._collection.count())


if __name__ == "__main__":
    main()

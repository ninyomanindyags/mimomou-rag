"""
Demo showcase untuk keperluan screenshot Bab 3 / Bab 4 skripsi.

TUJUAN: nunjukin proses chunking, embedding, dan retrieval dengan
CONTOH KECIL (bukan seluruh data), biar bisa langsung di-screenshot
dan ditempel ke laporan.

Cara pakai:
    (.venv) > python demo_showcase.py

Kamu bisa comment/uncomment bagian STEP di bawah sesuai bab mana yang
lagi kamu tulis (misal cuma mau screenshot chunking dulu -> comment
STEP 2 & STEP 3 dulu).
"""
import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from src.ingestion.loader import load_pdfs
from src.chunking.chunker import chunk_documents
from src.embeddings.embedder import load_embedder


def sep(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def preview(text, max_chars=350):
    text = text.strip().replace("\n", " ")
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


# ============================================================
# STEP 1: CHUNKING -- tampilkan proses split dokumen jadi chunk
# ============================================================
sep("STEP 1: CHUNKING")

documents = load_pdfs()
print(f"Jumlah dokumen (PDF) yang dimuat : {len(documents)}")

chunks = chunk_documents(documents)
print(f"Jumlah chunk hasil splitting     : {len(chunks)}")
print(f"Ukuran chunk (chunk_size)        : lihat config.yaml -> chunking.chunk_size")
print(f"Overlap antar chunk              : lihat config.yaml -> chunking.chunk_overlap")

print("\nContoh 2 chunk pertama (sample saja, bukan semua):\n")
for i, c in enumerate(chunks[:2]):
    print(f"--- Chunk #{i} ---")
    print(f"Metadata : {c.metadata}")
    print(f"Isi      : {preview(c.page_content)}")
    print()


# ============================================================
# STEP 2: EMBEDDING -- tampilkan hasil embedding 1 chunk sample
# ============================================================
sep("STEP 2: EMBEDDING")

sample_chunk = chunks[0]
embedder = load_embedder()
vector = embedder.embed_query(sample_chunk.page_content)

print(f"Model embedding      : BAAI/bge-m3 (lihat config.yaml -> embedding.model_name)")
print(f"Dimensi vector       : {len(vector)}")
print(f"Contoh isi chunk     : {preview(sample_chunk.page_content, 150)}")
print(f"Cuplikan vector (10 nilai pertama dari {len(vector)}):")
print([round(v, 5) for v in vector[:10]])


# ============================================================
# STEP 3: RETRIEVAL -- tampilkan hasil pencarian untuk 1 query contoh
# ============================================================
sep("STEP 3: RETRIEVAL (contoh query)")

from src.vectordb.vector_store import load_vector_store

QUERY_CONTOH = "Apa itu blu by BCA Digital?"  # ganti sesuai contoh yang mau kamu tampilkan
MODE = "SCG"  # atau "Baseline", tinggal ganti buat bandingin

try:
    db = load_vector_store(MODE)
    results = db.similarity_search_with_score(QUERY_CONTOH, k=3)

    print(f"Query   : {QUERY_CONTOH}")
    print(f"Top-{len(results)} hasil retrieval:\n")
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"[{rank}] score={score:.4f}")
        print(f"    metadata : {doc.metadata}")
        print(f"    isi      : {preview(doc.page_content, 200)}")
        print()
except Exception as e:
    print(f"(Lewati step retrieval, sesuaikan dulu nama fungsi vector_store-nya: {e})")

print("\nSelesai. Screenshot bagian yang kamu perlukan dari output di atas.")
"""Load / build / persist Chroma vector store untuk mode Baseline, SCG,
SCG_CONTEXTUAL, dan SCG_CONTEXTUAL_SHORT.

BM25 hanya digunakan untuk mode SCG_CONTEXTUAL_SHORT.
"""

import pickle
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from src.embeddings.embedder import load_embedder
from src.utils.helpers import load_config


# ============================================================
# CHROMA DATABASE PATH
# ============================================================

def get_db_path(mode: str) -> str:
    """
    Mengambil path Chroma berdasarkan mode.
    """

    config = load_config()["paths"]

    if mode == "Baseline":
        return config["chroma_baseline"]

    elif mode == "SCG_CONTEXTUAL":
        return config["chroma_scg_contextual"]

    elif mode == "SCG_CONTEXTUAL_SHORT":
        return config["chroma_scg_contextual_short"]

    else:
        return config["chroma_scg"]


# ============================================================
# LOAD CHROMA VECTOR STORE
# ============================================================

def load_vector_store(mode: str) -> Chroma:
    """
    Load koleksi Chroma yang sudah ada untuk mode tertentu.
    """

    db_path = get_db_path(mode)

    return Chroma(
        persist_directory=db_path,
        embedding_function=load_embedder(),
    )


# ============================================================
# BM25 PATH
# ============================================================

def get_bm25_path(mode: str) -> str:
    """
    Path BM25 index.

    BM25 hanya digunakan untuk mode
    SCG_CONTEXTUAL_SHORT dan disimpan di dalam
    folder Chroma mode tersebut.
    """

    if mode != "SCG_CONTEXTUAL_SHORT":
        raise ValueError(
            "BM25 hanya digunakan untuk mode "
            "SCG_CONTEXTUAL_SHORT."
        )

    return str(
        Path(get_db_path(mode)) / "bm25.pkl"
    )


# ============================================================
# RESET VECTOR STORE
# ============================================================

def reset_vector_store(mode: str):
    """
    Menghapus folder Chroma lama untuk mode tertentu,
    jika folder tersebut sudah ada.

    Karena BM25 SCG_CONTEXTUAL_SHORT disimpan di
    dalam folder yang sama, BM25 juga otomatis terhapus.
    """

    db_path = get_db_path(mode)

    if Path(db_path).exists():
        shutil.rmtree(db_path)


# ============================================================
# BUILD BM25 INDEX
# ============================================================

def build_bm25_index(documents, mode: str):
    """
    Membuat BM25 index dari documents.

    BM25 hanya dibuat untuk mode SCG_CONTEXTUAL_SHORT.

    Documents yang digunakan adalah contextualized documents,
    yaitu synthetic context + original chunk.
    """

    if mode != "SCG_CONTEXTUAL_SHORT":
        return

    print(
        f"\n[BM25] Building index untuk {mode}..."
    )

    # --------------------------------------------------------
    # Tokenisasi dokumen
    # --------------------------------------------------------

    tokenized_corpus = [
        document.page_content.split()
        for document in documents
    ]

    # --------------------------------------------------------
    # Build BM25
    # --------------------------------------------------------

    bm25 = BM25Okapi(tokenized_corpus)

    # --------------------------------------------------------
    # Simpan metadata untuk mapping index BM25
    # ke chunk_id
    # --------------------------------------------------------

    doc_metadata = [
        {
            "chunk_id": document.metadata.get("chunk_id"),
            "metadata": document.metadata,
        }
        for document in documents
    ]

    # --------------------------------------------------------
    # Simpan BM25 ke dalam folder Chroma
    # --------------------------------------------------------

    bm25_path = get_bm25_path(mode)

    Path(bm25_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(bm25_path, "wb") as file:
        pickle.dump(
            {
                "bm25": bm25,
                "documents": documents,
                "metadata": doc_metadata,
            },
            file,
        )

    print(
        f"[BM25] Index saved: "
        f"{bm25_path} "
        f"({len(documents)} docs)"
    )


# ============================================================
# LOAD BM25 INDEX
# ============================================================

def load_bm25_index(mode: str):
    """
    Load BM25 index untuk SCG_CONTEXTUAL_SHORT.
    """

    if mode != "SCG_CONTEXTUAL_SHORT":
        raise ValueError(
            "BM25 hanya tersedia untuk mode "
            "SCG_CONTEXTUAL_SHORT."
        )

    bm25_path = get_bm25_path(mode)

    if not Path(bm25_path).exists():
        raise FileNotFoundError(
            f"BM25 index tidak ditemukan: {bm25_path}. "
            f"Jalankan kembali build script "
            f"untuk {mode}."
        )

    with open(bm25_path, "rb") as file:
        data = pickle.load(file)

    return (
        data["bm25"],
        data["documents"],
        data["metadata"],
    )


# ============================================================
# PERSIST DOCUMENTS
# ============================================================

def persist_documents(documents, mode: str) -> Chroma:
    """
    Embed documents lalu simpan sebagai koleksi Chroma baru.

    Untuk SCG_CONTEXTUAL_SHORT:
        1. Documents di-embed dan disimpan ke Chroma.
        2. Documents yang sama digunakan untuk membangun BM25.

    Untuk mode lainnya:
        Hanya Chroma yang dibuat.
    """

    db_path = get_db_path(mode)

    # --------------------------------------------------------
    # Build Chroma
    # --------------------------------------------------------

    db = Chroma.from_documents(
        documents=documents,
        embedding=load_embedder(),
        persist_directory=db_path,
        collection_metadata={
            "hnsw:space": "cosine"
        },
    )

    # --------------------------------------------------------
    # Build BM25 hanya untuk SCG_CONTEXTUAL_SHORT
    # --------------------------------------------------------

    if mode == "SCG_CONTEXTUAL_SHORT":
        build_bm25_index(
            documents,
            mode,
        )

    return db
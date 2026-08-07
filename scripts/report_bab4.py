import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.vectordb.vector_store import load_vector_store
from src.embeddings.embedder import load_embedder

import yaml

from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
)
from langchain_core.documents import Document

try:
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
    )
except ImportError:
    from langchain.text_splitter import (
        RecursiveCharacterTextSplitter,
    )

# ============================================================
# PATH PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
PDF_DIR = PROJECT_ROOT / "data" / "pdf"

BASELINE_DB_DIR = (
    PROJECT_ROOT / "vectorstore" / "chroma_db"
)

SCG_DB_DIR = (
    PROJECT_ROOT / "vectorstore" / "chroma_db_scg"
)

SCG_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "vectorstore"
    / "scg_checkpoint.jsonl"
)

# ============================================================
# FUNGSI BANTU
# ============================================================

def print_title(title: str):
    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)

def clean_text(text: str) -> str:
    """
    Menyesuaikan fungsi preprocessing pada project.
    """

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def preview_text(text: str, limit: int = 800) -> str:
    """
    Menampilkan teks tanpa menghapus newline.
    Digunakan agar perbedaan sebelum dan sesudah
    preprocessing terlihat pada terminal.
    """

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    if len(text) > limit:
        return text[:limit] + "..."

    return text

def shorten_text(text: str, limit: int = 800) -> str:
    """
    Menampilkan teks dalam satu baris untuk contoh
    chunk dan synthetic context.
    """

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    if len(text) > limit:
        return text[:limit] + "..."

    return text

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        print(
            f"[WARNING] config.yaml tidak ditemukan: "
            f"{CONFIG_PATH}"
        )
        return {}

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file) or {}

def get_config_section(
    config: dict,
    *section_names: str,
) -> dict:
    """
    Mengambil bagian konfigurasi dengan beberapa
    kemungkinan nama key.
    """

    for section_name in section_names:
        section = config.get(section_name)

        if isinstance(section, dict):
            return section

    return {}

# ============================================================
# KONFIGURASI
# ============================================================

def report_config(config: dict):
    print_title("KONFIGURASI KNOWLEDGE BASE")

    chunking_config = get_config_section(
        config,
        "chunking",
    )

    embedding_config = get_config_section(
        config,
        "embedding",
        "embeddings",
    )

    retrieval_config = get_config_section(
        config,
        "retrieval",
    )

    reranker_config = get_config_section(
        config,
        "reranker",
    )

    paths_config = get_config_section(
        config,
        "paths",
    )

    print(f"Project root       : {PROJECT_ROOT}")
    print(f"Folder PDF         : {PDF_DIR}")
    print(f"Baseline database  : {BASELINE_DB_DIR}")
    print(f"SCG database       : {SCG_DB_DIR}")
    print(f"SCG checkpoint     : {SCG_CHECKPOINT_PATH}")

    print(
        "\nchunk_size         : "
        f"{chunking_config.get('chunk_size', 'tidak ditemukan')}"
    )

    print(
        "chunk_overlap      : "
        f"{chunking_config.get('chunk_overlap', 'tidak ditemukan')}"
    )

    print(
        "embedding model    : "
        f"{embedding_config.get('model_name', 'tidak ditemukan')}"
    )

    print(
        "reranker model     : "
        f"{reranker_config.get('model_name', 'tidak ditemukan')}"
    )

    print(
        "fetch_k_baseline   : "
        f"{retrieval_config.get('fetch_k_baseline', 'tidak ditemukan')}"
    )

    print(
        "fetch_k_scg        : "
        f"{retrieval_config.get('fetch_k_scg', 'tidak ditemukan')}"
    )

    print(
        "target_k           : "
        f"{retrieval_config.get('target_k', 'tidak ditemukan')}"
    )

    print(
        "path PDF config    : "
        f"{paths_config.get('data_pdf', 'tidak ditemukan')}"
    )

# ============================================================
# PEMUATAN DOKUMEN PDF
# ============================================================

def load_pdf_documents():
    """
    Membaca dokumen PDF menggunakan loader yang sama
    dengan proses indexing.

    Fungsi ini hanya membaca dokumen dan tidak mengubah
    file maupun database.
    """

    print_title("HASIL PEMUATAN DOKUMEN PDF")

    if not PDF_DIR.exists():
        print(
            f"[ERROR] Folder PDF tidak ditemukan: "
            f"{PDF_DIR}"
        )
        return []

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    print(f"Lokasi PDF         : {PDF_DIR}")
    print(f"Jumlah file PDF    : {len(pdf_files)}")

    if not pdf_files:
        print("[ERROR] Tidak ada file PDF.")
        return []

    try:
        loader = PyPDFDirectoryLoader(
            str(PDF_DIR)
        )

        documents = loader.load()

    except Exception as error:
        print(
            f"[ERROR] Gagal memuat dokumen PDF: "
            f"{error}"
        )
        return []

    print(
        "Jumlah Document "
        f"hasil loader      : {len(documents)}"
    )

    statistics = {}

    for document in documents:
        source = document.metadata.get(
            "source",
            "tidak diketahui",
        )

        source_name = Path(source).name

        if source_name not in statistics:
            statistics[source_name] = {
                "pages": 0,
                "characters": 0,
            }

        statistics[source_name]["pages"] += 1
        statistics[source_name]["characters"] += len(
            document.page_content
        )

    for number, pdf_path in enumerate(
        pdf_files,
        start=1,
    ):
        file_statistics = statistics.get(
            pdf_path.name,
            {
                "pages": 0,
                "characters": 0,
            },
        )

        print(
            f"{number:02d}. {pdf_path.name} | "
            f"halaman={file_statistics['pages']} | "
            f"karakter={file_statistics['characters']}"
        )

    return documents

# ============================================================
# PREPROCESSING
# ============================================================

def report_preprocessing(documents):
    print_title("HASIL PREPROCESSING DOKUMEN")

    if not documents:
        print(
            "[Tidak ada dokumen yang dapat ditampilkan]"
        )
        return

    document = documents[0]

    source = document.metadata.get(
        "source",
        "tidak diketahui",
    )

    raw_text = document.page_content
    cleaned_text = clean_text(raw_text)

    print(
        f"Contoh sumber: "
        f"{Path(source).name}"
    )

    print("\nSEBELUM PREPROCESSING:")
    print(preview_text(raw_text))

    print("\nSETELAH PREPROCESSING:")
    print(preview_text(cleaned_text))

    print("\nSTATISTIK:")
    print(
        f"Panjang sebelum       : "
        f"{len(raw_text)} karakter"
    )

    print(
        f"Panjang sesudah       : "
        f"{len(cleaned_text)} karakter"
    )

    print(
        f"Jumlah newline sebelum: "
        f"{raw_text.count(chr(10))}"
    )

    print(
        f"Jumlah newline sesudah: "
        f"{cleaned_text.count(chr(10))}"
    )

# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    documents,
    config: dict,
):
    """
    Mengulang proses preprocessing dan chunking
    untuk memperoleh statistik hasil aktual.

    Fungsi ini tidak membuat database dan tidak
    memanggil LLM.
    """

    chunking_config = get_config_section(
        config,
        "chunking",
    )

    chunk_size = chunking_config.get(
        "chunk_size",
        500,
    )

    chunk_overlap = chunking_config.get(
        "chunk_overlap",
        100,
    )

    cleaned_documents = []

    for document in documents:
        cleaned_document = Document(
            page_content=clean_text(
                document.page_content
            ),
            metadata=dict(document.metadata),
        )

        cleaned_documents.append(
            cleaned_document
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(
        cleaned_documents
    )

    return chunks

def report_chunking(
    chunks,
    config: dict,
):
    print_title("HASIL CHUNKING DOKUMEN")

    chunking_config = get_config_section(
        config,
        "chunking",
    )

    chunk_size = chunking_config.get(
        "chunk_size",
        500,
    )

    chunk_overlap = chunking_config.get(
        "chunk_overlap",
        100,
    )

    print(f"chunk_size        : {chunk_size}")
    print(f"chunk_overlap     : {chunk_overlap}")
    print(f"Jumlah total chunk: {len(chunks)}")

    if not chunks:
        print("[Tidak ada chunk]")
        return

    lengths = [
        len(chunk.page_content)
        for chunk in chunks
    ]

    average_length = (
        sum(lengths) / len(lengths)
    )

    print(
        f"Rata-rata panjang : "
        f"{average_length:.2f} karakter"
    )

    print(
        f"Panjang minimum   : "
        f"{min(lengths)} karakter"
    )

    print(
        f"Panjang maksimum  : "
        f"{max(lengths)} karakter"
    )

    print("\nCONTOH CHUNK:")

    for number, chunk in enumerate(
        chunks[:3],
        start=1,
    ):
        source = chunk.metadata.get(
            "source",
            "tidak diketahui",
        )

        print(f"\nChunk {number}")
        print(f"Sumber: {Path(source).name}")
        print(
            shorten_text(
                chunk.page_content,
                600,
            )
        )

# ============================================================
# HASIL PEMBENTUKAN EMBEDDING
# ============================================================

def report_embedding(
    database_dir: Path,
    database_name: str,
):
    print_title(f"HASIL PEMBENTUKAN EMBEDDING {database_name}")

    if not database_dir.exists():
        print(f"[WARNING] Database tidak ditemukan: {database_dir}")
        return

    sqlite_file = find_sqlite_file(database_dir)

    if sqlite_file is None:
        print("[WARNING] File sqlite tidak ditemukan.")
        return

    try:
        connection = sqlite3.connect(str(sqlite_file))
        cursor = connection.cursor()

        cursor.execute("SELECT COUNT(*) FROM embeddings")
        embedding_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT id) FROM embeddings")
        unique_embedding = cursor.fetchone()[0]

        print(f"Database           : {database_name}")
        print(f"Jumlah embedding   : {embedding_count}")
        print(f"Embedding unik     : {unique_embedding}")

        db = load_vector_store(database_name)

        data = db._collection.get(
            limit=1,
            include=["documents"]
        )

        if data["documents"]:
            embedder = load_embedder()

            embedding = embedder.embed_query(data["documents"][0])

            print(f"Dimensi embedding  : {len(embedding)}")
            print(f"5 nilai pertama    : {embedding[:5]}")

        connection.close()

    except Exception as error:
        print(f"[ERROR] {error}")

# ============================================================
# CHECKPOINT SYNTHETIC CONTEXT
# ============================================================

def report_scg_checkpoint():
    print_title(
        "HASIL PEMBENTUKAN SYNTHETIC CONTEXT"
    )

    if not SCG_CHECKPOINT_PATH.exists():
        print(
            "[WARNING] File checkpoint tidak ditemukan: "
            f"{SCG_CHECKPOINT_PATH}"
        )
        return

    records = []
    invalid_records = 0

    with SCG_CHECKPOINT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
                records.append(record)

            except json.JSONDecodeError:
                invalid_records += 1

                print(
                    "[WARNING] JSON tidak valid "
                    f"pada baris {line_number}"
                )

    successful_records = []

    for record in records:
        synthetic_context = record.get(
            "synthetic_context",
            "",
        )

        if synthetic_context and str(
            synthetic_context
        ).strip():
            successful_records.append(record)

    print(
        f"Jumlah record checkpoint   : "
        f"{len(records)}"
    )

    print(
        f"Synthetic context berhasil : "
        f"{len(successful_records)}"
    )

    print(
        f"JSON tidak valid            : "
        f"{invalid_records}"
    )

    if records:
        success_percentage = (
            len(successful_records)
            / len(records)
            * 100
        )

        print(
            f"Persentase keberhasilan    : "
            f"{success_percentage:.2f}%"
        )

    if not successful_records:
        return

    example = successful_records[0]

    chunk_id = example.get(
        "chunk_id",
        example.get("no", "-"),
    )

    print("\nCONTOH SYNTHETIC CONTEXT:")
    print(f"chunk_id: {chunk_id}")
    print(
        shorten_text(
            str(
                example.get(
                    "synthetic_context",
                    "",
                )
            ),
            1500,
        )
    )

# ============================================================
# PEMERIKSAAN FILE SQLITE CHROMADB
# ============================================================

def find_sqlite_file(database_dir: Path):
    possible_files = [
        database_dir / "chroma.sqlite3",
        database_dir / "chroma.sqlite",
    ]

    for file_path in possible_files:
        if file_path.exists():
            return file_path

    sqlite3_files = list(
        database_dir.glob("*.sqlite3")
    )

    if sqlite3_files:
        return sqlite3_files[0]

    sqlite_files = list(
        database_dir.glob("*.sqlite")
    )

    if sqlite_files:
        return sqlite_files[0]

    return None

def report_chromadb(
    database_dir: Path,
    database_name: str,
):
    print_title(
        f"HASIL CHROMADB {database_name}"
    )

    if not database_dir.exists():
        print(
            f"[WARNING] Folder database tidak ditemukan: "
            f"{database_dir}"
        )
        return

    sqlite_file = find_sqlite_file(
        database_dir
    )

    if sqlite_file is None:
        print(
            f"[WARNING] File SQLite tidak ditemukan "
            f"di {database_dir}"
        )
        return

    print(f"Folder database: {database_dir}")
    print(f"File SQLite    : {sqlite_file}")

    try:
        connection = sqlite3.connect(
            str(sqlite_file)
        )

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """
        )

        tables = [
            row[0]
            for row in cursor.fetchall()
        ]

        print("\nTabel database:")
        print(", ".join(tables))

        if "embeddings" in tables:
            cursor.execute(
                "SELECT COUNT(*) FROM embeddings"
            )

            embedding_count = cursor.fetchone()[0]

            print(
                "\nJumlah record embeddings: "
                f"{embedding_count}"
            )

        if "embedding_metadata" in tables:
            cursor.execute(
                """
                SELECT key, string_value, int_value
                FROM embedding_metadata
                WHERE key IN (
                    'source_type',
                    'chunk_id'
                )
                """
            )

            metadata_rows = cursor.fetchall()

            source_types = Counter()
            chunk_ids = set()

            for (
                key,
                string_value,
                int_value,
            ) in metadata_rows:
                if string_value is not None:
                    value = string_value
                else:
                    value = str(int_value)

                if key == "source_type":
                    source_types[value] += 1

                elif key == "chunk_id":
                    chunk_ids.add(value)

            print(
                "Distribusi source_type: "
                f"{dict(source_types)}"
            )

            print(
                "Jumlah chunk_id unik  : "
                f"{len(chunk_ids)}"
            )

        # ========================================================
        # CONTOH ISI CHROMADB
        # ========================================================

        db = load_vector_store(database_name)

        data = db._collection.get(
            limit=2,
            include=["documents", "metadatas"],
        )

        print("\nContoh isi ChromaDB:")

        for i, (doc, meta) in enumerate(
            zip(data["documents"], data["metadatas"]),
            start=1,
        ):
            print("-" * 80)
            print(f"Document {i}")
            print(f"Chunk ID    : {meta.get('chunk_id')}")
            print(f"Source      : {Path(meta.get('source', '')).name}")
            print(f"Source Type : {meta.get('source_type')}")
            print("Content:")
            print(shorten_text(doc, 500))

        connection.close()

    except Exception as error:
        print(
            f"[ERROR] Gagal membaca ChromaDB: "
            f"{error}"
        )

# ============================================================
# MAIN
# ============================================================

def main():
    print_title(
        "LAPORAN HASIL BAB 4 MIMOMOU"
    )

    config = load_config()

    report_config(config)

    documents = load_pdf_documents()

    report_preprocessing(documents)

    chunks = create_chunks(
        documents,
        config,
    )

    report_chunking(
        chunks,
        config,
    )

    report_embedding(
        BASELINE_DB_DIR,
        "BASELINE",
    )

    report_embedding(
        SCG_DB_DIR,
        "SCG",
    )

    report_scg_checkpoint()

    report_chromadb(
        BASELINE_DB_DIR,
        "BASELINE",
    )

    report_chromadb(
        SCG_DB_DIR,
        "SCG",
    )

    print_title("SELESAI")

    print(
        "Laporan selesai."
    )

    print(
        "Database dan checkpoint tidak diubah."
    )

if __name__ == "__main__":
    main()
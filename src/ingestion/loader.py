"""Load dokumen PDF dari folder data."""
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader

from src.utils.helpers import load_config


def load_pdfs(data_path: str | None = None):
    """
    Load semua PDF dari `data_path` (default: config.paths.data_pdf).

    BUG FIX: sebelumnya kalau folder PDF belum ada / kosong,
    PyPDFDirectoryLoader akan diam-diam menghasilkan list kosong, yang
    baru ketahuan error-nya jauh di bawah (chunking kosong -> Chroma
    crash dengan pesan yang membingungkan). Sekarang di-cek lebih awal
    dengan pesan yang jelas.
    """
    config = load_config()
    data_path = data_path or config["paths"]["data_pdf"]

    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"Folder PDF '{data_path}' tidak ditemukan. "
            f"Pastikan file PDF sudah diletakkan di folder tersebut."
        )

    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()

    if not documents:
        raise ValueError(f"Tidak ada PDF yang berhasil dibaca dari '{data_path}'.")

    return documents

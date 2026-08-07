"""Load dokumen PDF dari folder data."""
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader #  buat baca semua PDF di folder

from src.utils.helpers import load_config


def clean_text(text: str) -> str:
    """
    Membersihkan hasil ekstraksi PDF dengan:
    - mengubah newline menjadi spasi,
    - menghapus whitespace yang berlebihan.
    """
    text = text.replace("\n", " ") # ganti karakter enter jd spaso
    text = re.sub(r"\s+", " ", text) # mengubah whitespace berlebihan jd 1 spasi
    return text.strip() # mengubah spasi di awal/akhir


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

    # buat cek apakah folder PDF ada/tidak
    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"Folder PDF '{data_path}' tidak ditemukan. "
            f"Pastikan file PDF sudah diletakkan di folder tersebut."
        )

    # baca semua PDF
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()

    # cek apakah ada dokumen yang berhasil dibaca
    if not documents:
        raise ValueError(
            f"Tidak ada PDF yang berhasil dibaca dari '{data_path}'."
        )

    # membersihkan hasil ekstraksi PDF
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    return documents
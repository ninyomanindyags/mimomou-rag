"""Loader untuk LLM client (dipakai untuk generation, contextualize, dan SCG)."""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.utils.helpers import load_config

load_dotenv()

_llm = None
_scg_llm = None


def _build_llm() -> ChatOpenAI:
    config = load_config()["llm"]
    api_key = os.getenv("API_KEY")

    # BUG FIX: sebelumnya nggak ada pengecekan API_KEY -- kalau .env belum
    # diisi, error baru muncul jauh di bawah (dari dalam httpx/OpenAI client)
    # dengan pesan yang membingungkan. Sekarang gagal cepat dengan pesan jelas.
    if not api_key:
        raise EnvironmentError(
            "API_KEY tidak ditemukan. Isi API_KEY=... di file .env sebelum menjalankan aplikasi."
        )

    return ChatOpenAI(
        base_url=config["base_url"],
        api_key=api_key,
        model=config["model"],
        temperature=config["temperature"],
        request_timeout=config["request_timeout"],
        max_retries=config["max_retries"],
    )


def load_llm() -> ChatOpenAI:
    """LLM untuk generation jawaban chatbot + contextualize pertanyaan lanjutan.

    BUG FIX: sebelumnya function ini membuat instance ChatOpenAI BARU setiap
    kali dipanggil (tidak ada caching), padahal dipanggil ulang tiap
    pertanyaan -- boros dan tidak konsisten dengan pola singleton yang
    sudah dipakai reranker.py. Sekarang di-cache seperti singleton.
    """
    global _llm
    if _llm is None:
        _llm = _build_llm()
    return _llm


def load_scg_llm() -> ChatOpenAI:
    """LLM untuk generate synthetic context saat indexing (SCG).

    Dipisah dari load_llm() (walau konfigurasinya sama saat ini) supaya ke
    depan bisa pakai model/setting berbeda tanpa mengubah pemanggilnya.
    """
    global _scg_llm
    if _scg_llm is None:
        _scg_llm = _build_llm()
    return _scg_llm

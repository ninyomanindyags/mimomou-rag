"""Loader untuk LLM client (dipakai untuk generation, contextualize, dan SCG)."""
import os # buat akses environment variable (API_KEY) dari file .env

from dotenv import load_dotenv # buat load file .env ke environment variable
from langchain_openai import ChatOpenAI # buat akses LLM OpenAI (ChatGPT) via API

from src.utils.helpers import load_config

load_dotenv() 

_llm = None # menyimpan objek LLM untuk chatbot
_scg_llm = None # menyimpan objek LLM untuk proses SCG
_judge_llm = None

def _build_llm() -> ChatOpenAI:
    config = load_config()["llm"] 
    api_key = os.getenv("API_KEY")

    print("=" * 50)
    print("BASE_URL :", config["base_url"])
    print("MODEL    :", config["model"])
    print("API_KEY  :", repr(api_key))
    print("=" * 50)

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
    """
    Mengambil objek LLM untuk chatbot.
    Jika belum pernah dibuat, maka dibuat terlebih dahulu.
    """

    global _llm
    # lazy loading: objek LLM baru dibuat jika belum ada
    if _llm is None: 
        _llm = _build_llm() 
    # singleton: gunakan objek LLM yang sama untuk pemanggilan berikutnya
    return _llm

# ini dipake buat di build scg db, indexing --> digunakan saat proses indexing SCG untuk menghasilkan synthetic context setiap chunk.
def load_scg_llm() -> ChatOpenAI:
    """LLM untuk generate synthetic context saat indexing (SCG).

    Dipisah dari load_llm() (walau konfigurasinya sama saat ini) supaya ke
    depan bisa pakai model/setting berbeda tanpa mengubah pemanggilnya.
    """
    global _scg_llm
     # lazy loading: objek LLM baru dibuat jika belum ada
    if _scg_llm is None:
        _scg_llm = _build_llm()
    # singleton: gunakan objek LLM yang sama untuk pemanggilan berikutnya
    return _scg_llm

def load_judge_llm() -> ChatOpenAI:
    """LLM khusus sebagai evaluator/judge untuk RAGAS."""
    global _judge_llm

    if _judge_llm is None:
        config = load_config()["llm"]
        api_key = os.getenv("API_KEY")

        if not api_key:
            raise EnvironmentError(
                "API_KEY tidak ditemukan. Isi API_KEY=... di file .env."
            )

        _judge_llm = ChatOpenAI(
            base_url=config["base_url"],
            api_key=api_key,
            model=config["judge_model"],
            temperature=0,
            request_timeout=180,
            max_retries=3,
        )

    return _judge_llm

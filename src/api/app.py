"""FastAPI entry point untuk FinChat -- expose RAG chain lewat HTTP API.

Menggantikan main.py (CLI) sebagai cara menjalankan chatbot, supaya bisa
dipakai frontend web (HTML/CSS/JS). Logic RAG (init_rag_chain / ask) di
src/api/routes.py SAMA SEKALI TIDAK DIUBAH -- cuma dibungkus endpoint HTTP
di file ini.

Jalankan dengan:
    uvicorn src.api.app:app --reload
"""
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.api.routes import init_rag_chain
from src.prompts.prompt_templates import OPENING_MESSAGE
from src.vectordb.vector_store import get_db_path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

app = FastAPI(title="FinChat API")

# Izinkan frontend akses API ini (berguna kalau frontend dijalankan dari
# server/port berbeda saat development, misal Live Server VSCode).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache chain per mode ("Baseline" / "SCG"), supaya vector store + LLM
# cuma di-load sekali walau dipanggil berkali-kali (load-nya berat).
_chains: dict[str, object] = {}

# session_id -> chat_history (list of {"role": ..., "content": ...})
_sessions: dict[str, list] = {}


def _vector_store_exists(mode: str) -> bool:
    """Cek apakah folder ChromaDB untuk `mode` sudah ada isinya."""
    path = Path(get_db_path(mode))
    return path.exists() and any(path.iterdir())


def _ensure_vector_store(mode: str):
    """Build ChromaDB dari PDF kalau belum ada -- dipanggil sekali per mode,
    lazy (cuma pas mode itu pertama kali dipakai), bukan di setiap startup,
    supaya mode yang tidak dipakai tidak ikut memperlambat cold start."""
    if _vector_store_exists(mode):
        logging.info("Vector store %s sudah ada, skip build.", mode)
        return

    logging.info("Vector store %s belum ada -- membangun dari PDF (bisa lama)...", mode)
    if mode == "Baseline":
        from scripts.build_baseline_db import main as build_baseline
        build_baseline()
    else:
        from scripts.build_scg_db import main as build_scg
        build_scg()
    logging.info("Vector store %s selesai dibangun.", mode)


def get_chain(mode: str):
    if mode not in ("Baseline", "SCG"):
        raise HTTPException(status_code=400, detail="mode harus 'Baseline' atau 'SCG'")
    if mode not in _chains:
        _ensure_vector_store(mode)
        _chains[mode] = init_rag_chain(mode)
    return _chains[mode]


class ChatRequest(BaseModel):
    session_id: str
    message: str
    mode: str = "SCG"


class ChatResponse(BaseModel):
    answer: str
    session_id: str


@app.post("/api/session")
def new_session():
    """Dipanggil sekali di awal (saat halaman chat dibuka) buat dapat session_id."""
    session_id = str(uuid.uuid4())
    _sessions[session_id] = []
    return {"session_id": session_id, "opening_message": OPENING_MESSAGE.strip()}


@app.delete("/api/session/{session_id}")
def reset_session(session_id: str):
    """Reset history percakapan (dipakai tombol 'Chat Baru' di frontend)."""
    _sessions[session_id] = []
    return {"ok": True}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.session_id not in _sessions:
        # Frontend lupa/belum panggil /api/session -- tetap jalan, anggap sesi baru.
        _sessions[req.session_id] = []

    history = _sessions[req.session_id]
    ask = get_chain(req.mode)
    answer = ask(req.message, history)

    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": answer})
    logging.info("[%s] Q: %s | A: %s", req.mode, req.message, answer)

    return ChatResponse(answer=answer, session_id=req.session_id)


# Serve frontend statis (index.html/style.css/script.js) dari server yang
# sama, biar cukup jalanin satu server aja (tidak perlu mikirin CORS lagi).
# Harus di-mount PALING BAWAH, setelah semua route /api/... didefinisikan.
FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

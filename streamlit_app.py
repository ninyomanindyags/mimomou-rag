"""Entry point Streamlit -- UI chat untuk pipeline RAG MimoMou.

Streamlit digunakan sebagai antarmuka chatbot untuk memilih mode retrieval
dan menerima pertanyaan pengguna. Logic RAG (retrieval, generation,
fallback, dan error handling) tetap ditangani oleh src/api/routes.py.
"""

import logging
from pathlib import Path

import streamlit as st

from src.api.routes import init_rag_chain

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

st.set_page_config(
    page_title="MimoMou",
    page_icon="\U0001F4B0",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def get_ask_fn(mode: str):
    """Menginisialisasi RAG chain sesuai mode retrieval."""
    return init_rag_chain(mode)


# ============================================================
# HEADER
# ============================================================

st.title("💰 MimoMou")
st.caption("Chatbot edukasi literasi keuangan digital")


# ============================================================
# MODE RETRIEVAL
# ============================================================

mode = st.radio(
    "Mode retrieval",
    ["SCG", "Baseline"],
    index=0,
    horizontal=True,
)


# ============================================================
# LOAD RAG
# ============================================================

ask = get_ask_fn(mode)


# ============================================================
# CHAT
# ============================================================

question = st.chat_input("Tulis pertanyaanmu di sini...")

if question:
    st.chat_message("user").markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("MimoMou sedang mengetik..."):
            answer = ask(question)

        st.markdown(answer)

    logging.info(
        "[%s] Q: %s | A: %s",
        mode,
        question,
        answer,
    )
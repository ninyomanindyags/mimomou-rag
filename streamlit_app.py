"""Entry point Streamlit -- UI chat untuk pipeline RAG MimoMou.

Wrapper tipis di atas src.api.routes.init_rag_chain(), paralel dengan
main.py (versi CLI). Logic RAG (retrieval, generation, fallback, error
handling) tidak diduplikasi di sini -- semuanya tetap ada di
src/api/routes.py, file ini cuma menangani state & tampilan chat.
"""
import logging
from pathlib import Path

import streamlit as st

from src.api.routes import init_rag_chain
from src.prompts.prompt_templates import OPENING_MESSAGE

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
    """init_rag_chain() me-load vector store + LLM -- lumayan berat,
    jadi di-cache per mode supaya cuma dijalankan sekali per sesi server,
    bukan setiap kali user kirim pertanyaan / Streamlit rerun."""
    return init_rag_chain(mode)


def reset_chat():
    st.session_state.messages = []


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


# Reset histori otomatis kalau mode diganti, biar histori percakapan
# (dipakai untuk contextualize_question) tidak mencampur konteks dari
# dua mode yang berbeda.
if st.session_state.get("mode") != mode:
    st.session_state.mode = mode
    reset_chat()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# TOMBOL OBROLAN BARU
# ============================================================

st.button(
    "🔄 Mulai obrolan baru",
    on_click=reset_chat,
    use_container_width=True,
)


# ============================================================
# LOAD RAG
# ============================================================

ask = get_ask_fn(mode)


# ============================================================
# CHAT
# ============================================================

if not st.session_state.messages:
    st.chat_message("assistant").markdown(OPENING_MESSAGE)


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])


question = st.chat_input("Tulis pertanyaanmu di sini...")


if question:
    st.chat_message("user").markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("MimoMou sedang mengetik..."):
            answer = ask(question, st.session_state.messages)

        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    logging.info(
        "[%s] Q: %s | A: %s",
        mode,
        question,
        answer,
    )
"""Core chatbot logic: bangun RAG chain dan expose fungsi ask().

Dinamai api/routes.py supaya sesuai struktur folder proyek. Kalau nanti
proyek ini dipasangi FastAPI/Flask, endpoint HTTP tinggal dibuat di atas
init_rag_chain()/ask() di sini, tanpa perlu menyentuh logic RAG-nya.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from src.prompts.prompt_templates import SYSTEM_PROMPT, CONTEXTUALIZE_PROMPT
from src.llm.llm_client import load_llm
from src.vectordb.vector_store import load_vector_store
from src.retrieval.retriever import retrieve_docs
from src.utils.helpers import format_docs, load_config

FALLBACK_MESSAGE = "Maaf, informasi tersebut belum tersedia pada basis pengetahuan yang saya miliki."
ERROR_MESSAGE = "Maaf, sistem sedang mengalami gangguan koneksi ke layanan AI. Silakan coba beberapa saat lagi."


def init_rag_chain(mode: str):
    db = load_vector_store(mode)
    chat_model = load_llm()
    output_parser = StrOutputParser()
    max_history_turns = load_config()["conversation"]["max_history_turns"]

    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Pertanyaan: {question}"),
        ]
    )
    chain = chat_template | chat_model | output_parser

    def build_history_messages(chat_history):
        """
        chat_history: list of dict [{"role": "user"/"assistant", "content": "..."}]
        Diambil dari session_state Streamlit. Hanya N pasang terakhir yang dipakai.
        """
        if not chat_history:
            return []

        recent = chat_history[-(max_history_turns * 2):]
        messages = []
        for msg in recent:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        return messages

    def contextualize_question(question, chat_history):
        """
        Ubah pertanyaan lanjutan (yang butuh history buat dipahami) jadi
        pertanyaan mandiri, supaya tahap retrieval bisa cari dokumen yang
        relevan.
        """
        if not chat_history:
            return question

        recent = chat_history[-(max_history_turns * 2):]
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent)

        prompt = CONTEXTUALIZE_PROMPT.invoke({"history": history_text, "question": question})

        try:
            return chat_model.invoke(prompt).content.strip()
        except Exception as e:
            print(f"[Contextualize Error] {e}")
            return question

    def ask(question, chat_history=None):
        # BUG FIX: sebelumnya pertanyaan kosong/whitespace lolos ke
        # retrieval & LLM (buang-buang panggilan API untuk hasil yang
        # sudah pasti fallback). Sekarang di-cek lebih awal.
        if not question or not question.strip():
            return FALLBACK_MESSAGE

        standalone_question = contextualize_question(question, chat_history)

        docs = retrieve_docs(db, mode, standalone_question)

        if not docs:
            return FALLBACK_MESSAGE

        context = format_docs(docs)

        if not context.strip():
            return FALLBACK_MESSAGE

        history_messages = build_history_messages(chat_history)

        try:
            return chain.invoke(
                {"context": context, "question": question, "history": history_messages}
            )
        except Exception as e:
            print(f"[Generation Error] {e}")
            return ERROR_MESSAGE

    return ask

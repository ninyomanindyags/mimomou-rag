"""Core chatbot logic: bangun RAG chain dan expose fungsi ask()."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from src.prompts.prompt_templates import SYSTEM_PROMPT
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
        Mengubah riwayat percakapan menjadi format pesan LangChain.
        Hanya sejumlah percakapan terakhir yang digunakan sesuai konfigurasi.
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

    def ask(question, chat_history=None):
        # Cegah pertanyaan kosong
        if not question or not question.strip():
            return FALLBACK_MESSAGE

        docs = retrieve_docs(
            db=db,
            mode=mode,
            question=question,
        )

        if not docs:
            return FALLBACK_MESSAGE

        context = format_docs(docs)

        if not context.strip():
            return FALLBACK_MESSAGE

        history_messages = build_history_messages(chat_history)

        try:
            return chain.invoke(
                {
                    "context": context,
                    "question": question,
                    "history": history_messages,
                }
            )
        except Exception as e:
            print(f"[Generation Error] {e}")
            return ERROR_MESSAGE

    return ask
"""Entry point proyek -- CLI chat loop sederhana di atas pipeline RAG MimoMou.

Untuk versi asli (Streamlit) tinggal buat app.py yang import
`init_rag_chain` dari src.api.routes, sama seperti yang dipakai di sini.
"""
import logging
from pathlib import Path

from src.api.routes import init_rag_chain
from src.prompts.prompt_templates import OPENING_MESSAGE

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main(mode: str = "SCG"):
    print(OPENING_MESSAGE)
    ask = init_rag_chain(mode)
    chat_history = []

    while True:
        question = input("\nKamu: ").strip()
        if question.lower() in {"exit", "quit", "keluar"}:
            print("Sampai jumpa!")
            break
        if not question:
            continue

        answer = ask(question, chat_history)
        print(f"\nMimoMou: {answer}")

        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})
        logging.info("Q: %s | A: %s", question, answer)


if __name__ == "__main__":
    main()

"""Unit test dasar (sanity check) untuk modul-modul inti -- tanpa perlu
download model/embedding, biar cepat dijalankan di CI/lokal."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document

from src.utils.helpers import load_config, format_docs
from src.retrieval.retriever import rerank


def test_load_config_has_required_keys():
    config = load_config()
    for section in ["embedding", "llm", "chunking", "retrieval", "reranker", "paths", "conversation"]:
        assert section in config


def test_format_docs_joins_page_content():
    docs = [Document(page_content="halo"), Document(page_content="dunia")]
    assert format_docs(docs) == "halo\n\ndunia"


def test_format_docs_empty_list():
    assert format_docs([]) == ""


def test_rerank_empty_docs_returns_empty():
    assert rerank("pertanyaan apapun", []) == []


if __name__ == "__main__":
    test_load_config_has_required_keys()
    test_format_docs_joins_page_content()
    test_format_docs_empty_list()
    test_rerank_empty_docs_returns_empty()
    print("Semua test lulus.")

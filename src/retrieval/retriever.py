import time

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.utils.helpers import load_config

_reranker = None


def load_reranker():
    global _reranker

    if _reranker is None:
        config = load_config()["reranker"]
        _reranker = CrossEncoder(config["model_name"])

    return _reranker


def rerank(query, documents, top_n):
    if not documents:
        return []

    try:
        model = load_reranker()

        pairs = [
            (query, document.page_content)
            for document in documents
        ]

        scores = model.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        return [
            document
            for document, _ in ranked[:top_n]
        ]

    except Exception as error:
        print(
            f"[Reranker Error] {error}. "
            "Menggunakan urutan similarity search."
        )
        return documents[:top_n]


# Fungsi ini tetap dipertahankan.
# Tidak dipakai pada eksperimen ini, tetapi jangan dihapus
# supaya nanti bisa kembali ke pipeline lama dengan mudah.
def get_original_documents_by_chunk_ids(
    db,
    chunk_ids,
):
    if not chunk_ids:
        return []

    try:
        data = db.get(
            where={
                "source_type": "original",
            },
            include=[
                "documents",
                "metadatas",
            ],
        )

    except Exception as error:
        print(f"[Mapping Error] {error}")
        return []

    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    original_by_chunk_id = {}

    for content, metadata in zip(
        documents,
        metadatas,
    ):
        if not metadata:
            continue

        chunk_id = metadata.get("chunk_id")

        if chunk_id is None:
            continue

        original_by_chunk_id[str(chunk_id)] = Document(
            page_content=content,
            metadata={
                **metadata,
                "source_type": "original",
                "retrieval_source_type": "synthetic",
            },
        )

    selected_documents = []
    used_chunk_ids = set()

    for chunk_id in chunk_ids:
        key = str(chunk_id)

        if key in used_chunk_ids:
            continue

        document = original_by_chunk_id.get(key)

        if document is None:
            continue

        used_chunk_ids.add(key)
        selected_documents.append(document)

    return selected_documents


def print_results(title, results):
    print(f"\n{title}")

    for index, item in enumerate(results, start=1):
        if isinstance(item, tuple):
            document, score = item
            print(
                f"{index}. "
                f"chunk_id={document.metadata.get('chunk_id')} | "
                f"type={document.metadata.get('source_type')} | "
                f"score={float(score):.4f}"
            )
        else:
            document = item
            print(
                f"{index}. "
                f"chunk_id={document.metadata.get('chunk_id')} | "
                f"type={document.metadata.get('source_type')}"
            )


def retrieve_docs(
    db,
    mode,
    question,
):
    config = load_config()["retrieval"]

    fetch_key = (
        "fetch_k_scg"
        if mode == "SCG"
        else "fetch_k_baseline"
    )

    fetch_k = config[fetch_key]
    target_k = config["target_k"]

    source_type = (
        "synthetic"
        if mode == "SCG"
        else "original"
    )

    print("\n" + "=" * 60)
    print(f"MODE     : {mode}")
    print(f"QUESTION : {question}")
    print(f"FETCH_K  : {fetch_k}")
    print(f"TARGET_K : {target_k}")
    print("=" * 60)

    start = time.perf_counter()

    try:
        similarity_results = (
            db.similarity_search_with_relevance_scores(
                question,
                k=fetch_k,
                filter={
                    "source_type": source_type,
                },
            )
        )

    except Exception as error:
        elapsed = time.perf_counter() - start

        print(f"[Retrieval Error] {error}")
        print(
            f"[SIMILARITY TIME] {mode}: "
            f"{elapsed:.4f} detik"
        )

        return []

    elapsed = time.perf_counter() - start

    print(
        f"[SIMILARITY TIME] {mode}: "
        f"{elapsed:.4f} detik"
    )

    if not similarity_results:
        print("[Retrieval] Tidak ada hasil.")
        return []

    print_results(
        f"HASIL SIMILARITY SEARCH ({mode})",
        similarity_results,
    )

    candidates = [
        document
        for document, _ in similarity_results
    ]

    reranked_documents = rerank(
        question,
        candidates,
        top_n=target_k,
    )

    # ==========================
    # BASELINE
    # ==========================
    if mode != "SCG":
        print_results(
            "HASIL RERANK BASELINE - ORIGINAL UNTUK LLM",
            reranked_documents,
        )

        return reranked_documents

    # ==========================
    # SCG (VERSI EKSPERIMEN)
    # Synthetic langsung dikirim ke LLM
    # ==========================

    print_results(
        "HASIL RERANK SCG - SYNTHETIC UNTUK LLM",
        reranked_documents,
    )

    final_documents = []

    for doc in reranked_documents:
        final_documents.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "retrieval_source_type": "synthetic",
                },
            )
        )

    return final_documents
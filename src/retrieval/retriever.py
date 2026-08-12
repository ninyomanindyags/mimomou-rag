import time

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.utils.helpers import load_config


_reranker = None


# ============================================================
# LOAD RERANKER
# ============================================================

def load_reranker():
    global _reranker

    if _reranker is None:
        config = load_config()["reranker"]
        _reranker = CrossEncoder(config["model_name"])

    return _reranker


# ============================================================
# RERANKING
# ============================================================

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


# ============================================================
# MAPPING SYNTHETIC -> ORIGINAL
# ============================================================
#
# Digunakan pada mode SCG.
#
# Synthetic context digunakan untuk:
# 1. similarity search
# 2. reranking
#
# Setelah synthetic terpilih, chunk_id digunakan untuk
# mengambil kembali original chunk yang sesuai.
#
# Original chunk inilah yang nantinya dikirim ke LLM.
# ============================================================

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


# ============================================================
# PRINT RESULTS
# ============================================================

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


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_docs(
    db,
    mode,
    question,
):
    config = load_config()["retrieval"]

    # --------------------------------------------------------
    # FETCH K
    # --------------------------------------------------------
    #
    # Baseline:
    # similarity search dilakukan terhadap ORIGINAL.
    #
    # SCG:
    # similarity search dilakukan terhadap SYNTHETIC.
    # --------------------------------------------------------

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
    print(f"SOURCE   : {source_type}")
    print("=" * 60)

    # ========================================================
    # SIMILARITY SEARCH
    # ========================================================

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

    # --------------------------------------------------------
    # HASIL SIMILARITY SEARCH
    # --------------------------------------------------------

    print_results(
        f"HASIL SIMILARITY SEARCH ({mode})",
        similarity_results,
    )

    # ========================================================
    # RERANKING
    # ========================================================

    candidates = [
        document
        for document, _ in similarity_results
    ]

    reranked_documents = rerank(
        question,
        candidates,
        top_n=target_k,
    )

    # ========================================================
    # DEBUG HASIL RERANKING
    # ========================================================

    print("\n" + "=" * 80)
    print(f"HASIL RERANKING - {mode}")
    print("=" * 80)

    for i, doc in enumerate(
        reranked_documents,
        start=1,
    ):
        print(
            f"\n===== Document {i} ====="
        )

        print(
            f"chunk_id    : "
            f"{doc.metadata.get('chunk_id')}"
        )

        print(
            f"source_type : "
            f"{doc.metadata.get('source_type')}"
        )

        print("-" * 80)

        print(doc.page_content)

    # ========================================================
    # DEBUG KE FILE
    # ========================================================

    with open(
        "debug_retrieval.txt",
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            f"\n\nQUESTION: {question}\n"
        )

        f.write(
            f"MODE: {mode}\n"
        )

        f.write(
            f"FETCH_K: {fetch_k}\n"
        )

        f.write(
            f"TARGET_K: {target_k}\n"
        )

        f.write("=" * 80 + "\n")

        for i, doc in enumerate(
            reranked_documents,
            start=1,
        ):
            f.write(
                f"\n===== Document {i} =====\n"
            )

            f.write(
                f"chunk_id    : "
                f"{doc.metadata.get('chunk_id')}\n"
            )

            f.write(
                f"source_type : "
                f"{doc.metadata.get('source_type')}\n\n"
            )

            f.write(doc.page_content)
            f.write("\n")

    # ========================================================
    # BASELINE
    # ========================================================
    #
    # Baseline:
    #
    # original
    #    ↓
    # similarity search
    #    ↓
    # fetch_k kandidat
    #    ↓
    # reranking
    #    ↓
    # target_k
    #    ↓
    # original ke LLM
    #
    # ========================================================

    if mode != "SCG":

        print_results(
            "FINAL CONTEXT BASELINE - ORIGINAL UNTUK LLM",
            reranked_documents,
        )

        print("\n" + "=" * 80)
        print("FINAL CONTEXT UNTUK LLM - BASELINE")
        print("=" * 80)

        for i, doc in enumerate(
            reranked_documents,
            start=1,
        ):
            print(
                f"\n===== Document {i} ====="
            )

            print(
                f"chunk_id    : "
                f"{doc.metadata.get('chunk_id')}"
            )

            print(
                f"source_type : "
                f"{doc.metadata.get('source_type')}"
            )

            print("-" * 80)

            print(doc.page_content)

        return reranked_documents

    # ========================================================
    # SCG
    # ========================================================
    #
    # Untuk sementara jangan jadikan ini fokus.
    #
    # Alur:
    #
    # synthetic
    #    ↓
    # similarity search
    #    ↓
    # reranking
    #    ↓
    # chunk_id
    #    ↓
    # mapping ke original
    #    ↓
    # original ke LLM
    #
    # ========================================================

    print_results(
        "HASIL RERANK SCG - SYNTHETIC",
        reranked_documents,
    )

    # --------------------------------------------------------
    # AMBIL CHUNK ID
    # --------------------------------------------------------

    chunk_ids = [
        doc.metadata.get("chunk_id")
        for doc in reranked_documents
        if doc.metadata.get("chunk_id") is not None
    ]

    print("\n" + "=" * 80)
    print("MAPPING SCG SYNTHETIC -> ORIGINAL")
    print("=" * 80)

    print(
        f"Chunk ID terpilih: {chunk_ids}"
    )

    # --------------------------------------------------------
    # MAPPING KE ORIGINAL
    # --------------------------------------------------------

    original_documents = (
        get_original_documents_by_chunk_ids(
            db=db,
            chunk_ids=chunk_ids,
        )
    )

    # ========================================================
    # FINAL CONTEXT SCG
    # ========================================================

    print_results(
        "FINAL CONTEXT SCG - ORIGINAL UNTUK LLM",
        original_documents,
    )

    print("\n" + "=" * 80)
    print("FINAL CONTEXT UNTUK LLM - SCG")
    print("=" * 80)

    for i, doc in enumerate(
        original_documents,
        start=1,
    ):
        print(
            f"\n===== Document {i} ====="
        )

        print(
            f"chunk_id    : "
            f"{doc.metadata.get('chunk_id')}"
        )

        print(
            f"source_type : "
            f"{doc.metadata.get('source_type')}"
        )

        print("-" * 80)

        print(doc.page_content)

    return original_documents
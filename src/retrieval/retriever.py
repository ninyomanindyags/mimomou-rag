"""Retrieval untuk mode Baseline dan SCG_CONTEXTUAL_SHORT.

Baseline:
    Original chunk
    -> Semantic Search
    -> CrossEncoder Reranking
    -> Original chunk dikirim ke LLM

SCG_CONTEXTUAL_SHORT:
    Synthetic Context + Original Chunk
    -> Semantic Search + BM25
    -> RRF
    -> CrossEncoder Reranking
    -> Contextualized chunk dikirim ke LLM

BM25 dan RRF hanya digunakan pada mode SCG_CONTEXTUAL_SHORT.
"""

import time
from collections import defaultdict

from src.utils.helpers import load_config
from src.vectordb.vector_store import load_bm25_index

_reranker = None


# ============================================================
# LOAD RERANKER
# ============================================================

def load_reranker():
    """Load model CrossEncoder."""

    global _reranker

    if _reranker is None:
        from sentence_transformers import CrossEncoder

        config = load_config()["reranker"]
        _reranker = CrossEncoder(config["model_name"], device="cpu")    

    return _reranker


# ============================================================
# RERANKING
# ============================================================

def rerank(query, documents, top_n):
    """Reranking dokumen menggunakan CrossEncoder."""

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
            "Menggunakan urutan hasil retrieval."
        )

        return documents[:top_n]


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(title, results):
    """Menampilkan chunk ID dan score hasil retrieval."""

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if not results:
        print("Tidak ada hasil.")
        return

    for index, item in enumerate(results, start=1):

        if isinstance(item, tuple):
            document, score = item

            print(
                f"{index}. "
                f"chunk_id={document.metadata.get('chunk_id')} | "
                f"score={float(score):.4f}"
            )

        else:
            document = item

            print(
                f"{index}. "
                f"chunk_id={document.metadata.get('chunk_id')}"
            )


# ============================================================
# BM25 SEARCH
# ============================================================

def bm25_search(question, mode, k):
    """BM25 hanya digunakan pada SCG_CONTEXTUAL_SHORT."""

    if mode != "SCG_CONTEXTUAL_SHORT":
        return []

    try:
        bm25, documents, _ = load_bm25_index(mode)

        # Tokenisasi query
        tokenized_query = question.split()

        # Hitung BM25 score
        scores = bm25.get_scores(tokenized_query)

        # Ambil top-k dokumen
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        return [
            (documents[i], float(scores[i]))
            for i in top_indices
        ]

    except FileNotFoundError as error:
        print(f"[BM25 Warning] {error}")
        return []

    except Exception as error:
        print(f"[BM25 Error] {error}")
        return []


# ============================================================
# RRF FUSION
# ============================================================

def fuse_results(
    semantic_results,
    bm25_results,
    k_rrf=60,
):
    """
    Menggabungkan hasil Semantic Search dan BM25
    menggunakan Reciprocal Rank Fusion (RRF).

    RRF(d) = Σ 1 / (k + r(d))
    """

    fused_scores = defaultdict(float)
    documents = {}

    # --------------------------------------------------------
    # Semantic Search
    # --------------------------------------------------------

    for rank, (document, _) in enumerate(
        semantic_results,
        start=1,
    ):
        chunk_id = document.metadata.get("chunk_id")

        if chunk_id is None:
            continue

        fused_scores[chunk_id] += (
            1.0 / (k_rrf + rank)
        )

        documents[chunk_id] = document

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    for rank, (document, _) in enumerate(
        bm25_results,
        start=1,
    ):
        chunk_id = document.metadata.get("chunk_id")

        if chunk_id is None:
            continue

        fused_scores[chunk_id] += (
            1.0 / (k_rrf + rank)
        )

        documents[chunk_id] = document

    # --------------------------------------------------------
    # Sort berdasarkan skor RRF
    # --------------------------------------------------------

    ranked_chunk_ids = sorted(
        fused_scores.keys(),
        key=lambda chunk_id: fused_scores[chunk_id],
        reverse=True,
    )

    return [
        documents[chunk_id]
        for chunk_id in ranked_chunk_ids
    ]


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_docs(db, mode, question):
    """
    Retrieval berdasarkan mode.

    Baseline:
        Semantic Search
        -> CrossEncoder
        -> Original chunk

    SCG_CONTEXTUAL_SHORT:
        Semantic Search + BM25
        -> RRF
        -> CrossEncoder
        -> Contextualized chunk
    """

    config = load_config()["retrieval"]

    # ========================================================
    # CONFIGURATION
    # ========================================================

    if mode == "Baseline":

        fetch_k = config["fetch_k_baseline"]

    elif mode == "SCG_CONTEXTUAL_SHORT":

        fetch_k = config["fetch_k_scg"]

    else:

        raise ValueError(
            f"Mode tidak didukung: {mode}. "
            "Gunakan 'Baseline' atau "
            "'SCG_CONTEXTUAL_SHORT'."
        )

    target_k = config["target_k"]

    # ========================================================
    # START
    # ========================================================

    print("\n\n" + "#" * 80)
    print(f"RETRIEVAL START - {mode}")
    print("#" * 80)

    print(f"QUESTION : {question}")
    print(f"FETCH_K  : {fetch_k}")
    print(f"TARGET_K : {target_k}")

    # ========================================================
    # 1. SEMANTIC SEARCH
    # ========================================================

    print("\n" + "=" * 80)
    print("1. SEMANTIC SEARCH")
    print("=" * 80)

    start = time.perf_counter()

    try:
        semantic_results = (
            db.similarity_search_with_relevance_scores(
                question,
                k=fetch_k,
            )
        )

    except Exception as error:

        print(f"[Retrieval Error] {error}")

        return []

    elapsed = time.perf_counter() - start

    print(
        f"[SEMANTIC TIME] "
        f"{mode}: {elapsed:.4f} detik"
    )

    if not semantic_results:

        print(
            "[Retrieval] "
            "Tidak ada hasil semantic search."
        )

        return []

    print_results(
        "CHUNK HASIL SEMANTIC SEARCH",
        semantic_results,
    )

    # ========================================================
    # BASELINE
    # ========================================================

    if mode == "Baseline":

        # Baseline tidak menggunakan BM25 dan RRF.
        candidates = [
            document
            for document, _ in semantic_results
        ]

    # ========================================================
    # SCG_CONTEXTUAL_SHORT
    # ========================================================

    else:

        # ====================================================
        # 2. BM25 SEARCH
        # ====================================================

        print("\n" + "=" * 80)
        print("2. BM25 SEARCH")
        print("=" * 80)

        start = time.perf_counter()

        bm25_results = bm25_search(
            question,
            mode,
            k=fetch_k,
        )

        elapsed = time.perf_counter() - start

        print(
            f"[BM25 TIME] "
            f"{mode}: {elapsed:.4f} detik"
        )

        print_results(
            "CHUNK HASIL BM25",
            bm25_results,
        )

        # ====================================================
        # 3. RRF FUSION
        # ====================================================

        print("\n" + "=" * 80)
        print("3. RRF FUSION")
        print("=" * 80)

        if bm25_results:

            candidates = fuse_results(
                semantic_results,
                bm25_results,
            )

            print(
                "Semantic Search + BM25 "
                "digabungkan menggunakan RRF."
            )

        else:

            print(
                "BM25 tidak menghasilkan data. "
                "Menggunakan hasil Semantic Search."
            )

            candidates = [
                document
                for document, _ in semantic_results
            ]

        print_results(
            "CHUNK HASIL RRF",
            candidates,
        )

    # ========================================================
    # CROSSENCODER RERANKING
    # ========================================================

    step_number = (
        "2"
        if mode == "Baseline"
        else "4"
    )

    print("\n" + "=" * 80)
    print(f"{step_number}. CROSSENCODER RERANKING")
    print("=" * 80)

    start = time.perf_counter()

    final_documents = rerank(
        question,
        candidates,
        top_n=target_k,
    )

    elapsed = time.perf_counter() - start

    print(
        f"[RERANKER TIME] "
        f"{mode}: {elapsed:.4f} detik"
    )

    print_results(
        "CHUNK HASIL CROSSENCODER",
        final_documents,
    )

    # ========================================================
    # FINAL CONTEXT UNTUK LLM
    # ========================================================

    final_step = (
        "3"
        if mode == "Baseline"
        else "5"
    )

    print("\n" + "=" * 80)
    print(f"{final_step}. FINAL CONTEXT UNTUK LLM")
    print("=" * 80)

    for index, document in enumerate(
        final_documents,
        start=1,
    ):

        print(
            f"\n===== Document {index} ====="
        )

        print(
            f"chunk_id : "
            f"{document.metadata.get('chunk_id')}"
        )

        print("-" * 80)

        print(document.page_content)

    # ========================================================
    # DEBUG KE FILE
    # ========================================================

    with open(
        "debug_retrieval.txt",
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            f"\n\n{'#' * 80}\n"
        )

        file.write(
            f"MODE: {mode}\n"
        )

        file.write(
            f"QUESTION: {question}\n"
        )

        file.write(
            f"FETCH_K: {fetch_k}\n"
        )

        file.write(
            f"TARGET_K: {target_k}\n"
        )

        file.write(
            f"{'#' * 80}\n"
        )

        for index, document in enumerate(
            final_documents,
            start=1,
        ):

            file.write(
                f"\n===== Document {index} =====\n"
            )

            file.write(
                f"chunk_id: "
                f"{document.metadata.get('chunk_id')}\n\n"
            )

            file.write(
                document.page_content
            )

            file.write("\n")

    print(
        f"\n[FINAL] {len(final_documents)} "
        f"document dikirim ke LLM."
    )

    print("#" * 80)

    return final_documents
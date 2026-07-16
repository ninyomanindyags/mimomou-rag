"""Retrieval: similarity search + score filtering + dedup (SCG) + rerank.

Logic yang SAMA persis dipakai chatbot (src/api/routes.py) dan
scripts/evaluate_retrieval.py, supaya evaluasi RAGAS benar-benar
merepresentasikan pipeline yang dipakai user.
"""
import os

from sentence_transformers import CrossEncoder

from src.utils.helpers import load_config

_reranker = None


def load_reranker():
    """Reranker model, singleton -- model ini lumayan berat buat di-load ulang-ulang."""
    global _reranker
    if _reranker is None:
        config = load_config()["reranker"]
        _reranker = CrossEncoder(config["model_name"])
    return _reranker


def rerank(query, docs, top_n=10):
    """
    Urutkan ulang `docs` berdasarkan relevansi terhadap `query` pakai cross-encoder.
    Kalau reranker gagal, fallback ke urutan similarity search biar chatbot
    tetap jalan, nggak crash total.
    """
    if not docs:
        return []

    try:
        model = load_reranker()
        pairs = [(query, doc.page_content) for doc in docs]
        scores = model.predict(pairs)
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_n]]
    except Exception as e:
        print(f"[Reranker Error] {e} - fallback ke urutan similarity search")
        return docs[:top_n]


def retrieve_docs(db, mode: str, question: str):
    """
    db     : instance Chroma (hasil load_vector_store())
    mode   : "Baseline" atau "SCG"
    question : pertanyaan (idealnya standalone_question, sudah dikontekstualisasi)
    """
    config = load_config()["retrieval"]

    # SCORE_THRESHOLD tetap bisa dioverride lewat .env tanpa ubah kode/config.yaml
    score_threshold = float(os.getenv("SCORE_THRESHOLD", config["score_threshold"]))

    # fetch_k dinaikkan untuk KEDUA mode supaya reranker selalu punya kandidat
    # lebih banyak dari target_k untuk dipilih ulang. SCG lebih tinggi karena
    # tiap chunk_id punya 2 versi (original+synthetic) yang bakal kepotong
    # separuh pas dedup, jadi butuh kandidat mentah lebih banyak dari awal.
    fetch_k = config["fetch_k_scg"] if mode == "SCG" else config["fetch_k_baseline"]
    target_k = config["target_k"]

    try:
        results = db.similarity_search_with_relevance_scores(question, k=fetch_k)
    except Exception as e:
        print(f"[Retrieval Error] {e}")
        return []

    # Buang dokumen yang relevansinya terlalu rendah (nggak nyambung sama query)
    filtered = [(doc, score) for doc, score in results if score >= score_threshold]

    # Baseline: tidak ada pasangan original/synthetic, jadi tidak perlu dedup.
    if mode != "SCG":
        candidate_docs = [doc for doc, _ in filtered]
        return rerank(question, candidate_docs, top_n=target_k)

    # SCG: dedup berdasarkan chunk_id (bukan page!), hindari original & synthetic
    # dari chunk sumber yang sama sama-sama masuk top-k.
    seen = set()
    deduped_docs = []
    for doc, score in filtered:
        key = doc.metadata.get("chunk_id")
        if key in seen:
            continue
        seen.add(key)
        deduped_docs.append(doc)

    return rerank(question, deduped_docs, top_n=target_k)

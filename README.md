# MimoMou — RAG Chatbot Edukasi Bank Digital (Baseline vs SCG)

MimoMou adalah chatbot edukasi keuangan digital yang membandingkan dua
pendekatan retrieval dalam pipeline RAG (Retrieval-Augmented Generation):

- **Baseline** — similarity search biasa (naive RAG).
- **SCG (Synthetic Context Generation)** — setiap chunk diperkaya dengan
  synthetic context hasil generate LLM saat indexing, untuk meningkatkan
  kualitas retrieval.

Knowledge base mencakup tiga bank digital di Indonesia: **blu by BCA
Digital**, **Bank Jago**, dan **SeaBank**.

## Struktur Proyek

```
rag_project/
├── README.md
├── requirements.txt
├── .env.example        # salin jadi .env lalu isi API_KEY
├── .gitignore
├── config.yaml         # semua konfigurasi (model, chunking, threshold, path)
├── src/
│   ├── ingestion/       # load PDF -> loader.py
│   ├── chunking/        # split dokumen jadi chunk -> chunker.py
│   ├── embeddings/      # embedding model (BAAI/bge-m3) -> embedder.py
│   ├── vectordb/        # load/build/persist ChromaDB -> vector_store.py
│   ├── retrieval/       # similarity search + threshold + rerank -> retriever.py
│   ├── prompts/         # semua prompt template -> prompt_templates.py
│   ├── llm/             # loader LLM (Mistral via router.bynara.id) -> llm_client.py
│   ├── api/             # logic chatbot (init_rag_chain/ask) -> routes.py
│   └── utils/           # helper (config loader, checkpoint, format_docs) -> helpers.py
├── scripts/
│   ├── build_baseline_db.py   # build ChromaDB Baseline
│   ├── build_scg_db.py        # build ChromaDB SCG (dengan checkpoint)
│   ├── evaluate_retrieval.py  # jalankan RAGAS untuk Baseline vs SCG
│   └── test_threshold.py      # kalibrasi SCORE_THRESHOLD
├── tests/
│   └── test_app.py
├── logs/
│   └── app.log
├── data/pdf/            # taruh korpus PDF di sini
└── main.py               # entry point (CLI chat loop)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # lalu isi API_KEY
```

Taruh PDF korpus literasi keuangan di `data/pdf/`.

## Cara Pakai

1. **Build vector store:**
   ```bash
   python scripts/build_baseline_db.py
   python scripts/build_scg_db.py
   ```
2. **(Opsional) Kalibrasi ulang threshold:**
   ```bash
   python scripts/test_threshold.py
   ```
3. **Jalankan chatbot (CLI):**
   ```bash
   python main.py
   ```
4. **Evaluasi RAGAS (Baseline vs SCG):**
   ```bash
   python scripts/evaluate_retrieval.py
   ```

## Catatan Perubahan dari Versi Sebelumnya

Lihat komentar `BUG FIX` di masing-masing file untuk detail perbaikan
(API key check, singleton LLM/embedding, guard folder PDF kosong, dan
`build_scg_db.py` yang sekarang berhenti kalau ada chunk yang gagal
diproses, bukan diam-diam membuat DB SCG yang tidak lengkap).

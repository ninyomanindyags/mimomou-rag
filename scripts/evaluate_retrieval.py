"""Evaluasi retrieval + generation (RAGAS) untuk pipeline Baseline vs SCG."""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from src.llm.llm_client import load_llm
from src.embeddings.embedder import load_embedder
from src.vectordb.vector_store import load_vector_store
from src.retrieval.retriever import retrieve_docs
from src.utils.helpers import format_docs, load_checkpoint, append_checkpoint, load_config
from src.prompts.prompt_templates import SYSTEM_PROMPT

chat_model = load_llm()
config = load_config()
GT_PATH = config["paths"]["ground_truth"]
CHECKPOINT_PATH = config["paths"]["eval_checkpoint"]

EVAL_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Pertanyaan: {question}"),
    ]
)


def is_record_complete(rec):
    """
    Record dianggap belum selesai kalau salah satu jawaban None, kosong,
    atau context-nya kosong -- biar baris yang gagal generate bisa dicoba
    ulang di run berikutnya, bukan dianggap "selesai" selamanya.
    """
    if rec is None:
        return False

    ans_base = rec.get("answer_baseline")
    ans_scg = rec.get("answer_scg")
    ctx_base = rec.get("context_baseline")
    ctx_scg = rec.get("context_scg")

    if ans_base is None or ans_scg is None:
        return False
    if not str(ans_base).strip() or not str(ans_scg).strip():
        return False
    if not ctx_base or not ctx_scg:
        return False

    return True


def run_pipeline(question, db, mode, chat_model, max_retry=3):
    docs = retrieve_docs(db, mode, question)
    contexts = [d.page_content for d in docs]
    context_text = format_docs(docs) if docs else ""

    prompt = EVAL_PROMPT_TEMPLATE.invoke({"context": context_text, "question": question})

    answer = None
    for attempt in range(max_retry):
        try:
            answer = chat_model.invoke(prompt).content
            break
        except Exception as e:
            print(f"  retry {attempt + 1}/{max_retry} - {e}")
            time.sleep(5)

    return answer, contexts


def main():
    df = pd.read_excel(GT_PATH)
    done = load_checkpoint(CHECKPOINT_PATH, key_field="no")
    if done:
        print(f"Resume: {len(done)} baris ditemukan di checkpoint.\n")

    db_baseline = load_vector_store("Baseline")
    db_scg = load_vector_store("SCG")

    for idx, row in df.iterrows():
        no = int(row["No"])

        if no in done and is_record_complete(done[no]):
            continue

        if no in done and not is_record_complete(done[no]):
            print(f"[{no}] Hasil sebelumnya null/gagal, mencoba ulang...")

        question = row["Question"]
        print(f"[{no}/{len(df)}] {question}")

        ans_base, ctx_base = run_pipeline(question, db_baseline, "Baseline", chat_model)
        ans_scg, ctx_scg = run_pipeline(question, db_scg, "SCG", chat_model)

        rec = {
            "no": no,
            "answer_baseline": ans_base,
            "context_baseline": ctx_base,
            "answer_scg": ans_scg,
            "context_scg": ctx_scg,
        }
        append_checkpoint(CHECKPOINT_PATH, rec)
        done[no] = rec

    done = load_checkpoint(CHECKPOINT_PATH, key_field="no")

    still_failed = [no for no, rec in done.items() if not is_record_complete(rec)]
    if still_failed:
        print(f"\n⚠️  Masih ada {len(still_failed)} baris gagal setelah retry: {still_failed}")
        print("Jalankan ulang script ini sekali lagi untuk mencoba baris tersebut.")

    for no, rec in done.items():
        mask = df["No"] == no
        df.loc[mask, "Answer_Baseline"] = rec["answer_baseline"]
        df.loc[mask, "Answer_SCG"] = rec["answer_scg"]
        df.loc[mask, "Retrieved_Context_Baseline"] = "\n---\n".join(rec["context_baseline"]) if rec["context_baseline"] else ""
        df.loc[mask, "Retrieved_Context_SCG"] = "\n---\n".join(rec["context_scg"]) if rec["context_scg"] else ""

    df.to_excel(GT_PATH, index=False)
    print(f"\nSpreadsheet terupdate: {GT_PATH}")

    if still_failed:
        print("\nMasih ada baris gagal -- RAGAS TIDAK dijalankan dulu supaya hasil evaluasi tidak timpang.")
        print("Jalankan ulang script ini sampai semua baris lengkap, baru RAGAS akan otomatis jalan.")
        return

    # ------------------------ RAGAS EVALUATION ------------------------------
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig

    ragas_llm = LangchainLLMWrapper(chat_model)
    ragas_embeddings = LangchainEmbeddingsWrapper(load_embedder())
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    run_config = RunConfig(max_workers=1, timeout=90, max_retries=3)

    def build_dataset(answer_col, context_col):
        return Dataset.from_dict({
            "question": df["Question"].tolist(),
            "answer": [str(a) if pd.notna(a) else "" for a in df[answer_col].tolist()],
            "contexts": [str(c).split("\n---\n") if pd.notna(c) else [] for c in df[context_col].tolist()],
            "ground_truth": df["Ground_Truth"].tolist(),
        })

    print("\nMenjalankan RAGAS untuk pipeline BASELINE...")
    result_baseline = evaluate(
        build_dataset("Answer_Baseline", "Retrieved_Context_Baseline"),
        metrics=metrics, llm=ragas_llm, embeddings=ragas_embeddings,
        run_config=run_config,
    )

    print("Menjalankan RAGAS untuk pipeline SCG...")
    result_scg = evaluate(
        build_dataset("Answer_SCG", "Retrieved_Context_SCG"),
        metrics=metrics, llm=ragas_llm, embeddings=ragas_embeddings,
        run_config=run_config,
    )

    df_base = result_baseline.to_pandas()
    df_scg = result_scg.to_pandas()

    df_base.to_excel("ragas_result_baseline.xlsx", index=False)
    df_scg.to_excel("ragas_result_scg.xlsx", index=False)

    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    summary = pd.DataFrame({
        "Metric": metric_names,
        "Baseline": [df_base[m].mean() for m in metric_names],
        "SCG": [df_scg[m].mean() for m in metric_names],
    })
    summary["Selisih (SCG - Baseline)"] = summary["SCG"] - summary["Baseline"]
    summary.to_excel("ragas_summary_comparison.xlsx", index=False)

    print("\n=== RINGKASAN PERBANDINGAN ===")
    print(summary.to_string(index=False))
    print("\nFile hasil: ragas_result_baseline_scg.xlsx, ragas_result_scg_scg.xlsx, ragas_summary_comparison_scg.xlsx")


if __name__ == "__main__":
    main()

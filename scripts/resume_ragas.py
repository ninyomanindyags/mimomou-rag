"""Melanjutkan evaluasi RAGAS hanya pada sel metrik yang masih kosong."""

import ast
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "False"

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
os.chdir(ROOT)

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

from src.embeddings.embedder import load_embedder
from src.llm.llm_client import load_llm

METRICS = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
}

FILES = [
    ROOT / "ragas_result_baseline.xlsx",
    ROOT / "ragas_result_scg.xlsx",
]


def parse_contexts(value):
    if pd.isna(value) or not str(value).strip():
        return []

    text = str(value)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass

    return [part.strip() for part in text.split("\n---\n") if part.strip()]


def save_safely(df, path):
    temp_path = path.with_name(path.stem + ".resume_tmp.xlsx")
    df.to_excel(temp_path, index=False)
    temp_path.replace(path)


def evaluate_one(row, metric_names, ragas_llm, ragas_embeddings):
    dataset = Dataset.from_dict({
        "question": [str(row["user_input"])],
        "answer": [str(row["response"])],
        "contexts": [parse_contexts(row["retrieved_contexts"])],
        "ground_truth": [str(row["reference"])],
    })

    result = evaluate(
        dataset,
        metrics=[METRICS[name] for name in metric_names],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(
            max_workers=1,
            timeout=90,
            max_retries=3,
        ),
    )
    return result.to_pandas().iloc[0]


def resume_file(path, ragas_llm, ragas_embeddings):
    df = pd.read_excel(path)
    missing_by_row = {}

    for index, row in df.iterrows():
        missing = [
            name for name in METRICS
            if pd.isna(row.get(name))
        ]
        if missing:
            missing_by_row[index] = missing

    print(f"\nFile: {path.name}")
    print(f"Baris total: {len(df)}")
    print(f"Baris yang perlu dilanjutkan: {len(missing_by_row)}")

    if not missing_by_row:
        print("Tidak ada nilai kosong. File dilewati.")
        return

    backup = path.with_name(path.stem + ".before_resume.xlsx")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"Backup dibuat: {backup.name}")

    for number, (index, missing) in enumerate(missing_by_row.items(), start=1):
        question = str(df.at[index, "user_input"])
        print(f"[{number}/{len(missing_by_row)}] Baris Excel {index + 2}")
        print(f"  Metrik kosong: {', '.join(missing)}")
        print(f"  Pertanyaan: {question[:120]}")

        try:
            values = evaluate_one(
                df.loc[index],
                missing,
                ragas_llm,
                ragas_embeddings,
            )

            for metric in missing:
                value = values.get(metric)
                if pd.notna(value):
                    df.at[index, metric] = float(value)
                    print(f"  {metric}: {float(value):.6f}")
                else:
                    print(f"  {metric}: masih NaN")

            save_safely(df, path)
            print("  Disimpan.")

        except Exception as error:
            print(f"  Gagal: {error}")
            print("  Baris dipertahankan dan akan dicoba lagi pada run berikutnya.")

        time.sleep(1)

    remaining = int(df[list(METRICS)].isna().any(axis=1).sum())
    print(f"Selesai. Baris yang masih memiliki NaN: {remaining}")
    print(f"File diperbarui: {path}")


def main():
    print("Memuat model evaluasi...")
    chat_model = load_llm()
    embedder = load_embedder()
    ragas_llm = LangchainLLMWrapper(chat_model)
    ragas_embeddings = LangchainEmbeddingsWrapper(embedder)

    for path in FILES:
        if not path.exists():
            print(f"File tidak ditemukan, dilewati: {path}")
            continue
        resume_file(path, ragas_llm, ragas_embeddings)


if __name__ == "__main__":
    main()

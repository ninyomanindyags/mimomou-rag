"""Resume nilai NaN RAGAS tanpa menimpa workbook asli."""

import ast
import os
import shutil
import sys
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "False"

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.append(str(ROOT))

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
    (
        ROOT / "ragas_result_baseline.xlsx",
        ROOT / "ragas_result_baseline_resumed.xlsx",
    ),
    (
        ROOT / "ragas_result_scg.xlsx",
        ROOT / "ragas_result_scg_resumed.xlsx",
    ),
]


def parse_contexts(value):
    """Mengubah isi retrieved_contexts Excel menjadi list context."""
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


def atomic_save(df, output_path):
    temp_path = output_path.with_name(
        output_path.stem + "_temporary.xlsx"
    )
    df.to_excel(temp_path, index=False)
    temp_path.replace(output_path)


def evaluate_missing_metrics(
    row,
    missing_metrics,
    ragas_llm,
    ragas_embeddings,
):
    dataset = Dataset.from_dict({
        "question": [str(row["user_input"])],
        "answer": [str(row["response"])],
        "contexts": [parse_contexts(row["retrieved_contexts"])],
        "ground_truth": [str(row["reference"])],
    })

    result = evaluate(
        dataset,
        metrics=[METRICS[name] for name in missing_metrics],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(
            max_workers=1,
            timeout=90,
            max_retries=3,
        ),
    )

    return result.to_pandas().iloc[0]


def prepare_output(source_path, output_path):
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    if not output_path.exists():
        shutil.copy2(source_path, output_path)
        print(f"File baru dibuat: {output_path.name}")
    else:
        print(f"Melanjutkan file yang sudah ada: {output_path.name}")


def process_file(
    source_path,
    output_path,
    ragas_llm,
    ragas_embeddings,
):
    prepare_output(source_path, output_path)

    original_df = pd.read_excel(source_path)
    result_df = pd.read_excel(output_path)

    if len(original_df) != len(result_df):
        raise ValueError(
            f"Jumlah baris berubah pada {output_path.name}: "
            f"asli={len(original_df)}, hasil={len(result_df)}"
        )

    missing_rows = {}

    for index, row in result_df.iterrows():
        missing = [
            metric
            for metric in METRICS
            if pd.isna(row.get(metric))
        ]
        if missing:
            missing_rows[index] = missing

    print(f"\n=== {output_path.name} ===")
    print(f"Jumlah baris: {len(result_df)}")
    print(f"Baris yang memiliki NaN: {len(missing_rows)}")

    for number, (index, missing) in enumerate(
        missing_rows.items(),
        start=1,
    ):
        question = str(result_df.at[index, "user_input"])
        print(
            f"[{number}/{len(missing_rows)}] "
            f"baris Excel {index + 2}"
        )
        print(f"Metrik: {', '.join(missing)}")
        print(f"Pertanyaan: {question[:150]}")

        try:
            values = evaluate_missing_metrics(
                result_df.loc[index],
                missing,
                ragas_llm,
                ragas_embeddings,
            )

            for metric in missing:
                value = values.get(metric)

                if pd.notna(value):
                    result_df.at[index, metric] = float(value)
                    print(f"{metric}: {float(value):.6f}")
                else:
                    print(f"{metric}: masih NaN")

            # Simpan setiap selesai satu baris agar aman jika proses terhenti.
            atomic_save(result_df, output_path)
            print("Disimpan ke file hasil baru.")

        except Exception as error:
            print(f"Gagal memproses baris ini: {error}")
            print("Baris tetap kosong dan dapat dilanjutkan pada run berikutnya.")

    remaining = int(
        result_df[list(METRICS)].isna().any(axis=1).sum()
    )

    changed_existing = []
    newly_filled = []

    for index in range(len(original_df)):
        for metric in METRICS:
            old_value = original_df.at[index, metric]
            new_value = result_df.at[index, metric]

            if pd.isna(old_value) and not pd.isna(new_value):
                newly_filled.append((index + 2, metric, new_value))

            elif not pd.isna(old_value) and pd.isna(new_value):
                changed_existing.append(
                    (index + 2, metric, old_value, new_value)
                )

            elif (
                not pd.isna(old_value)
                and not pd.isna(new_value)
                and old_value != new_value
            ):
                changed_existing.append(
                    (index + 2, metric, old_value, new_value)
                )

    print(f"NaN berhasil diisi: {len(newly_filled)}")
    print(f"NaN yang masih tersisa: {remaining}")
    print(f"Nilai lama yang berubah: {len(changed_existing)}")

    if changed_existing:
        print("PERINGATAN: ada nilai lama yang berubah:")
        for item in changed_existing:
            print(item)
    else:
        print("OK: nilai yang sudah ada tidak berubah.")


def main():
    print("Memuat model LLM dan embedding untuk evaluasi RAGAS...")
    chat_model = load_llm()
    embedder = load_embedder()

    ragas_llm = LangchainLLMWrapper(chat_model)
    ragas_embeddings = LangchainEmbeddingsWrapper(embedder)

    for source_path, output_path in FILES:
        process_file(
            source_path,
            output_path,
            ragas_llm,
            ragas_embeddings,
        )

    print("\nSelesai.")
    print("File asli tidak ditimpa:")
    print("- ragas_result_baseline.xlsx")
    print("- ragas_result_scg.xlsx")
    print("File hasil baru:")
    print("- ragas_result_baseline_resumed.xlsx")
    print("- ragas_result_scg_resumed.xlsx")


if __name__ == "__main__":
    main()

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
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig

from src.llm.llm_client import load_llm
from src.embeddings.embedder import load_embedder


METRICS = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
}


FILES = {
    "baseline": (
        "ragas_result_baseline.xlsx",
        "ragas_result_baseline_resumed.xlsx",
    ),
    "scg": (
        "ragas_result_scg.xlsx",
        "ragas_result_scg_resumed.xlsx",
    ),
}


def get_contexts(value):

    if pd.isna(value):
        return []

    try:
        value = ast.literal_eval(str(value))

        if isinstance(value, list):
            return [str(x) for x in value]

    except Exception:
        pass

    return str(value).split("\n---\n")


def evaluate_row(row, missing, llm, embeddings):

    data = Dataset.from_dict({
        "question": [str(row["user_input"])],
        "answer": [str(row["response"])],
        "contexts": [
            get_contexts(row["retrieved_contexts"])
        ],
        "ground_truth": [str(row["reference"])],
    })

    result = evaluate(
        data,
        metrics=[METRICS[x] for x in missing],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(
            max_workers=3,
            timeout=300,
            max_retries=3,
        ),
    )

    return result.to_pandas().iloc[0]


def resume_file(name, llm, embeddings):

    source_name, output_name = FILES[name]

    source = ROOT / source_name
    output = ROOT / output_name

    # Jika file resumed belum ada, buat dari file asli.
    if not output.exists():

        shutil.copy2(source, output)

        print(f"\nMembuat file: {output_name}")

    else:

        print(f"\nMemperbaiki file: {output_name}")

    # HANYA file resumed yang akan diproses.
    df = pd.read_excel(output)

    print(f"Jumlah data: {len(df)}")

    total_nan_awal = int(
        df[list(METRICS)].isna().sum().sum()
    )

    print(f"Total NaN sebelum diperbaiki: {total_nan_awal}")

    # Proses setiap baris.
    for index, row in df.iterrows():

        # Cari hanya metrik yang masih NaN.
        missing = [
            metric
            for metric in METRICS
            if pd.isna(row[metric])
        ]

        # Kalau baris sudah lengkap, lewati.
        if not missing:
            continue

        print(
            f"\n{name.upper()} | "
            f"baris Excel {index + 2}"
        )

        print(
            f"Metrik yang diperbaiki: "
            f"{', '.join(missing)}"
        )

        try:

            values = evaluate_row(
                row,
                missing,
                llm,
                embeddings,
            )

            berhasil = []

            for metric in missing:

                value = values.get(metric)

                # Hanya masukkan kalau hasilnya benar-benar angka.
                if pd.notna(value):

                    df.loc[index, metric] = float(value)

                    berhasil.append(metric)

                    print(
                        f"  {metric}: "
                        f"{float(value):.6f}"
                    )

                else:

                    print(
                        f"  {metric}: masih NaN"
                    )

            # Simpan setelah setiap baris.
            df.to_excel(
                output,
                index=False
            )

            if berhasil:
                print(
                    f"  ✓ Tersimpan: "
                    f"{', '.join(berhasil)}"
                )

        except Exception as error:

            print(
                f"  ✗ Gagal: {error}"
            )

    # Cek kondisi akhir.
    total_nan_akhir = int(
        df[list(METRICS)].isna().sum().sum()
    )

    print("\n" + "=" * 60)
    print(f"SELESAI: {output_name}")
    print(f"NaN awal : {total_nan_awal}")
    print(f"NaN akhir: {total_nan_akhir}")
    print("=" * 60)


def main():

    print("Memuat model evaluasi...")

    llm = LangchainLLMWrapper(
        load_llm()
    )

    embeddings = LangchainEmbeddingsWrapper(
        load_embedder()
    )

    # Hanya memperbaiki file resumed.
    resume_file(
        "baseline",
        llm,
        embeddings
    )

    resume_file(
        "scg",
        llm,
        embeddings
    )

    print("\nFile asli tetap aman.")
    print("File yang diperbaiki:")
    print("- ragas_result_baseline_resumed.xlsx")
    print("- ragas_result_scg_resumed.xlsx")


if __name__ == "__main__":
    main()
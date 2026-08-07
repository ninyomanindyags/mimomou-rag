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
        "contexts": [get_contexts(row["retrieved_contexts"])],
        "ground_truth": [str(row["reference"])],
    })

    result = evaluate(
        data,
        metrics=[METRICS[x] for x in missing],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(
            max_workers=1,
            timeout=90,
            max_retries=3,
        ),
    )

    return result.to_pandas().iloc[0]

def resume_file(name, llm, embeddings):
    source_name, output_name = FILES[name]

    source = ROOT / source_name
    output = ROOT / output_name

    if not output.exists():
        shutil.copy2(source, output)
        print(f"\nMembuat: {output_name}")
    else:
        print(f"\nMelanjutkan: {output_name}")

    df_source = pd.read_excel(source)
    df = pd.read_excel(output)

    if len(df_source) != len(df):
        raise ValueError("Jumlah baris file sumber dan output berbeda.")

    for index, row in df.iterrows():
        missing = [
            metric
            for metric in METRICS
            if pd.isna(row[metric])
        ]

        if not missing:
            continue

        print(
            f"\n{name.upper()} | baris {index + 2} "
            f"| metrik: {', '.join(missing)}"
        )

        try:
            values = evaluate_row(
                row,
                missing,
                llm,
                embeddings,
            )

            for metric in missing:
                if pd.notna(values[metric]):
                    df.loc[index, metric] = float(
                        values[metric]
                    )
                    print(
                        f"{metric}: "
                        f"{float(values[metric]):.6f}"
                    )
                else:
                    print(f"{metric}: masih NaN")

            # Simpan setiap selesai satu baris.
            df.to_excel(output, index=False)

        except Exception as error:
            print(f"Gagal: {error}")

    print(f"\nSelesai: {output_name}")
    print(
        "NaN tersisa:",
        int(df[list(METRICS)].isna().sum().sum()),
    )

def main():
    print("Memuat model evaluasi...")
    llm = LangchainLLMWrapper(load_llm())
    embeddings = LangchainEmbeddingsWrapper(load_embedder())

    resume_file("baseline", llm, embeddings)
    resume_file("scg", llm, embeddings)

    print("\nFile asli tetap aman.")
    print("Output:")
    print("- ragas_result_baseline_resumed.xlsx")
    print("- ragas_result_scg_resumed.xlsx")

if __name__ == "__main__":
    main()
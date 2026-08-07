"""Menghitung ulang ringkasan RAGAS dari workbook hasil yang sudah lengkap.

Script ini TIDAK menjalankan ulang RAGAS. Script hanya membaca hasil
Baseline dan SCG yang sudah ada, menghitung rata-rata, serta selisih
rata-rata SCG - Baseline.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "ragas_result_baseline_resumed.xlsx"
SCG_PATH = ROOT / "ragas_result_scg_resumed.xlsx"
OUTPUT_PATH = ROOT / "ragas_summary_comparison_final.xlsx"

METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]
QUESTION_COLUMN = "user_input"


def main():
    baseline = pd.read_excel(BASELINE_PATH)
    scg = pd.read_excel(SCG_PATH)

    if len(baseline) != len(scg):
        raise ValueError(
            f"Jumlah baris tidak sama: Baseline={len(baseline)}, "
            f"SCG={len(scg)}."
        )

    if QUESTION_COLUMN not in baseline or QUESTION_COLUMN not in scg:
        raise ValueError(
            f"Kolom pasangan '{QUESTION_COLUMN}' tidak ditemukan "
            "pada salah satu workbook."
        )

    if not baseline[QUESTION_COLUMN].equals(scg[QUESTION_COLUMN]):
        raise ValueError(
            "Urutan pertanyaan Baseline dan SCG berbeda. "
            "Pastikan kedua workbook memakai pasangan pertanyaan yang sama."
        )

    missing_baseline = baseline[METRICS].isna().sum()
    missing_scg = scg[METRICS].isna().sum()

    if missing_baseline.sum() or missing_scg.sum():
        raise ValueError(
            "Masih ada nilai kosong/NaN. "
            f"Baseline={missing_baseline.to_dict()}, "
            f"SCG={missing_scg.to_dict()}."
        )

    baseline_mean = baseline[METRICS].mean()
    scg_mean = scg[METRICS].mean()

    summary = pd.DataFrame(
        {
            "Metric": METRICS,
            "Baseline": [baseline_mean[metric] for metric in METRICS],
            "SCG": [scg_mean[metric] for metric in METRICS],
        }
    )
    summary["Selisih (SCG - Baseline)"] = (
        summary["SCG"] - summary["Baseline"]
    )

    summary.to_excel(OUTPUT_PATH, index=False)

    print("=== RINGKASAN RAGAS DARI DATA LENGKAP ===")
    print(f"Jumlah pasangan pertanyaan: {len(baseline)}")
    print(summary.to_string(index=False))
    print(f"\nHasil disimpan di: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

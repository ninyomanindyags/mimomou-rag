"""
Menghitung ulang ringkasan RAGAS dari workbook hasil yang sudah lengkap.

Script ini TIDAK menjalankan ulang RAGAS.
Script hanya membaca hasil Baseline dan SCG, lalu menghitung:
- rata-rata Baseline
- rata-rata SCG
- selisih SCG - Baseline
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
    # Membaca workbook hasil RAGAS yang sudah lengkap
    baseline = pd.read_excel(BASELINE_PATH)
    scg = pd.read_excel(SCG_PATH)

    # Memastikan jumlah baris sama
    if len(baseline) != len(scg):
        raise ValueError(
            f"Jumlah baris tidak sama: "
            f"Baseline={len(baseline)}, SCG={len(scg)}."
        )

    # Memastikan kolom pertanyaan tersedia
    if (
        QUESTION_COLUMN not in baseline
        or QUESTION_COLUMN not in scg
    ):
        raise ValueError(
            f"Kolom '{QUESTION_COLUMN}' tidak ditemukan "
            "pada salah satu workbook."
        )

    # Memastikan pertanyaan Baseline dan SCG berpasangan
    if not baseline[QUESTION_COLUMN].equals(
        scg[QUESTION_COLUMN]
    ):
        raise ValueError(
            "Urutan pertanyaan Baseline dan SCG berbeda. "
            "Pastikan kedua workbook memakai pertanyaan "
            "yang sama dan urutannya sejajar."
        )

    # Memastikan tidak ada NaN
    missing_baseline = baseline[METRICS].isna().sum()
    missing_scg = scg[METRICS].isna().sum()

    if missing_baseline.sum() > 0:
        raise ValueError(
            "Baseline masih memiliki nilai NaN: "
            f"{missing_baseline.to_dict()}"
        )

    if missing_scg.sum() > 0:
        raise ValueError(
            "SCG masih memiliki nilai NaN: "
            f"{missing_scg.to_dict()}"
        )

    # Menghitung rata-rata setiap metrik
    baseline_mean = baseline[METRICS].mean()
    scg_mean = scg[METRICS].mean()

    summary = pd.DataFrame(
        {
            "Metric": METRICS,
            "Baseline": [
                baseline_mean[metric]
                for metric in METRICS
            ],
            "SCG": [
                scg_mean[metric]
                for metric in METRICS
            ],
        }
    )

    # Menghitung selisih rata-rata
    summary["Selisih (SCG - Baseline)"] = (
        summary["SCG"] - summary["Baseline"]
    )

    # Menyimpan hasil ke file baru
    summary.to_excel(OUTPUT_PATH, index=False)

    print("=== RINGKASAN RAGAS DATA LENGKAP ===")
    print(f"Jumlah pasangan pertanyaan: {len(baseline)}")
    print()
    print(summary.to_string(index=False))
    print()
    print(f"Hasil disimpan di: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
"""Uji Wilcoxon Signed-Rank untuk membandingkan hasil RAGAS
antara pipeline Baseline dan SCG."""

from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]

BASELINE_PATH = ROOT / "ragas_result_baseline.xlsx"
SCG_PATH = ROOT / "ragas_result_scg.xlsx"

OUTPUT_PATH = ROOT / "ragas_wilcoxon_test_testtt.xlsx"

QUESTION_COLUMN = "user_input"

METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


def run_wilcoxon(df_base, df_scg):
    results = []

    for metric in METRIC_NAMES:

        base = df_base[metric].astype(float)
        scg = df_scg[metric].astype(float)

        # Pastikan tidak ada nilai kosong
        if base.isna().any() or scg.isna().any():
            raise ValueError(
                f"Masih terdapat nilai NaN pada metrik '{metric}'. "
                "Lengkapi hasil evaluasi terlebih dahulu."
            )

        differences = scg - base

        # Wilcoxon tidak bisa dijalankan jika seluruh pasangan identik
        if (differences == 0).all():
            statistic = float("nan")
            p_value = float("nan")
            conclusion = "Seluruh pasangan identik"
        else:
            statistic, p_value = wilcoxon(
                base,
                scg,
                alternative="two-sided",
            )

            conclusion = (
                "Terdapat perbedaan signifikan"
                if p_value < 0.05
                else "Tidak terdapat perbedaan signifikan"
            )

        results.append(
            {
                "Metric": metric,
                "Jumlah_Pasangan": len(base),
                "Mean_Baseline": base.mean(),
                "Mean_SCG": scg.mean(),
                "Selisih_Rata2": differences.mean(),
                "Wilcoxon_Statistic": statistic,
                "p_value": p_value,
                "Kesimpulan": conclusion,
            }
        )

    return pd.DataFrame(results)


def main():

    df_base = pd.read_excel(BASELINE_PATH)
    df_scg = pd.read_excel(SCG_PATH)

    # Jumlah data harus sama
    if len(df_base) != len(df_scg):
        raise ValueError(
            f"Jumlah data tidak sama "
            f"(Baseline={len(df_base)}, SCG={len(df_scg)})."
        )

    # Pastikan kolom pertanyaan ada
    if QUESTION_COLUMN not in df_base.columns:
        raise ValueError(
            f"Kolom '{QUESTION_COLUMN}' tidak ditemukan "
            "pada file Baseline."
        )

    if QUESTION_COLUMN not in df_scg.columns:
        raise ValueError(
            f"Kolom '{QUESTION_COLUMN}' tidak ditemukan "
            "pada file SCG."
        )

    # Pastikan urutan pertanyaan sama
    if not df_base[QUESTION_COLUMN].equals(df_scg[QUESTION_COLUMN]):
        raise ValueError(
            "Urutan pertanyaan Baseline dan SCG berbeda. "
            "Wilcoxon harus menggunakan pasangan pertanyaan yang sama."
        )

    # Pastikan semua metrik tersedia
    missing_base = df_base[METRIC_NAMES].isna().sum()
    missing_scg = df_scg[METRIC_NAMES].isna().sum()

    if missing_base.sum() > 0 or missing_scg.sum() > 0:
        raise ValueError(
            "Masih terdapat nilai kosong.\n"
            f"Baseline : {missing_base.to_dict()}\n"
            f"SCG      : {missing_scg.to_dict()}"
        )

    result = run_wilcoxon(df_base, df_scg)

    result.to_excel(OUTPUT_PATH, index=False)

    print("=" * 60)
    print("HASIL UJI WILCOXON SIGNED-RANK")
    print("=" * 60)
    print(f"Jumlah pasangan data : {len(df_base)}")
    print(result.to_string(index=False))
    print(f"\nHasil disimpan pada:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
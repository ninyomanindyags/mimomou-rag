"""Kalibrasi SCORE_THRESHOLD: cari titik pisah antara skor query relevan vs irrelevan."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.vectordb.vector_store import load_vector_store, get_db_path

TEST_QUERIES = {
    "relevant": [
        "Aku sering melihat tips investasi di media sosial. Apa semua informasi tersebut bisa langsung dipercaya?",
        "Apa itu dana darurat, dan bedanya sama tabungan biasa?",
        "Kalau saya pinjam pinjol legal dengan tenor 4 bulan, berapa batas maksimal bunga per hari yang boleh dikenakan?",
        "Saya sudah punya dana darurat 3 bulan pengeluaran sebagai lajang, cukup gak menurut standar OJK, dan langkah apa yang seharusnya saya prioritaskan setelahnya investasi atau asuransi dulu?",
        "Bandingkan biaya Tidak Aktif Bank Jago (mulai berlaku Mei 2026) dengan kebijakan biaya SeaBank dan blu, bank mana yang punya risiko biaya tersembunyi kalau akun didiamkan lama?",
    ],
    "irrelevant": [
        "Bagaimana cara membuat rendang yang enak?",
        "Siapa pemenang piala dunia 2022?",
        "Apa itu fotosintesis?",
        "Bagaimana cara memperbaiki AC yang bocor?",
        "Rekomendasi film horor terbaik tahun ini",
    ],
}


def run_test(mode):
    db_path = get_db_path(mode)
    if not Path(db_path).exists():
        script = "build_scg_db.py" if mode == "SCG" else "build_baseline_db.py"
        print(f"\n[SKIP] Mode '{mode}': folder '{db_path}' belum ada. Jalankan scripts/{script} dulu.\n")
        return None, None

    print(f"\n{'=' * 60}")
    print(f"MODE: {mode}  (db: {db_path})")
    print(f"{'=' * 60}")

    db = load_vector_store(mode)

    all_relevant_scores = []
    all_irrelevant_scores = []

    for category, queries in TEST_QUERIES.items():
        print(f"\n--- Kategori: {category} ---")

        for q in queries:
            try:
                results = db.similarity_search_with_relevance_scores(q, k=3)
            except Exception as e:
                print(f"  [ERROR] '{q}': {e}")
                continue

            if not results:
                print(f"  '{q}' -> tidak ada hasil sama sekali")
                continue

            top_score = results[0][1]

            if category == "relevant":
                all_relevant_scores.append(top_score)
            else:
                all_irrelevant_scores.append(top_score)

            print(f"  '{q}'")
            for i, (doc, score) in enumerate(results, 1):
                preview = doc.page_content[:80].replace("\n", " ")
                print(f"      #{i} score={score:.4f} | {preview}...")

    if all_relevant_scores and all_irrelevant_scores:
        min_relevant = min(all_relevant_scores)
        max_irrelevant = max(all_irrelevant_scores)

        print(f"\n--- Ringkasan Mode {mode} ---")
        print(f"  Score relevant   -> min: {min_relevant:.4f}, "
              f"max: {max(all_relevant_scores):.4f}, "
              f"avg: {sum(all_relevant_scores)/len(all_relevant_scores):.4f}")
        print(f"  Score irrelevant -> min: {min(all_irrelevant_scores):.4f}, "
              f"max: {max_irrelevant:.4f}, "
              f"avg: {sum(all_irrelevant_scores)/len(all_irrelevant_scores):.4f}")

        if min_relevant > max_irrelevant:
            suggested = (min_relevant + max_irrelevant) / 2
            print(f"  ✅ Ada gap yang jelas. Saran SCORE_THRESHOLD ({mode} saja): {suggested:.4f}")
        else:
            print(f"  ⚠️  Ada overlap antara relevant & irrelevant score. "
                  f"Threshold nggak akan sempurna, pertimbangkan query test "
                  f"tambahan atau evaluasi manual per kasus.")

    return all_relevant_scores, all_irrelevant_scores


def print_combined_threshold(relevant_by_mode, irrelevant_by_mode):
    all_relevant = [s for scores in relevant_by_mode.values() for s in scores]
    all_irrelevant = [s for scores in irrelevant_by_mode.values() for s in scores]

    if not all_relevant or not all_irrelevant:
        print("\n[SKIP] Threshold gabungan tidak bisa dihitung -- salah satu mode belum punya data (db belum dibuat).")
        return

    min_relevant = min(all_relevant)
    max_irrelevant = max(all_irrelevant)

    print(f"\n{'=' * 60}")
    print("THRESHOLD GABUNGAN (Baseline + SCG)")
    print(f"{'=' * 60}")
    print(f"  Semua score relevant   -> min: {min_relevant:.4f}, max: {max(all_relevant):.4f}")
    print(f"  Semua score irrelevant -> min: {min(all_irrelevant):.4f}, max: {max_irrelevant:.4f}")

    if min_relevant > max_irrelevant:
        combined = (min_relevant + max_irrelevant) / 2
        print(f"  ✅ Ada gap yang jelas antar seluruh mode.")
        print(f"  👉 SCORE_THRESHOLD gabungan yang disarankan: {combined:.4f}")
        print(f"     (dipakai sama untuk Baseline maupun SCG, lihat .env / config.yaml -> retrieval.score_threshold)")
    else:
        print(f"  ⚠️  Ada overlap antara relevant & irrelevant score lintas mode. "
              f"Threshold tunggal mungkin kurang optimal untuk salah satu mode; "
              f"pertimbangkan menambah query test.")


if __name__ == "__main__":
    relevant_by_mode = {}
    irrelevant_by_mode = {}

    rel, irr = run_test("Baseline")
    if rel is not None:
        relevant_by_mode["Baseline"] = rel
        irrelevant_by_mode["Baseline"] = irr

    rel, irr = run_test("SCG")
    if rel is not None:
        relevant_by_mode["SCG"] = rel
        irrelevant_by_mode["SCG"] = irr

    print_combined_threshold(relevant_by_mode, irrelevant_by_mode)

import re
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
from langchain_community.document_loaders import PyPDFDirectoryLoader
from pypdf import PdfReader

# supaya bisa import module dari project
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.chunking.chunker import chunk_documents
from src.utils.helpers import load_config


# ============================================================
# PARAMETER CHUNKING YANG INGIN DIUJI
# ============================================================

TEST_PARAMS = [
    (300, 50),
    (500, 100),
    (800, 150),
    (1000, 200),
]


# ============================================================
# PREPROCESSING
# ============================================================

def clean_text(text: str) -> str:
    """
    Membersihkan hasil ekstraksi PDF dengan:
    - mengubah newline menjadi spasi,
    - menghapus whitespace yang berlebihan.
    """

    # Ubah newline menjadi spasi
    text = text.replace("\n", " ")

    # Ubah whitespace berlebihan menjadi satu spasi
    text = re.sub(r"\s+", " ", text)

    # Hapus spasi di awal dan akhir
    return text.strip()


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # LOAD CONFIG
    # ========================================================

    config = load_config()

    data_path = Path(config["paths"]["data_pdf"])

    if not data_path.exists():
        raise FileNotFoundError(
            f"Folder PDF '{data_path}' tidak ditemukan."
        )

    # ========================================================
    # CEK DOKUMEN PDF
    # ========================================================

    pdf_files = sorted(data_path.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(
            f"Tidak ada file PDF di folder '{data_path}'."
        )

    print(f"\nDitemukan {len(pdf_files)} dokumen PDF:\n")

    document_results = []
    total_pages = 0

    for i, pdf_file in enumerate(pdf_files, start=1):

        try:
            reader = PdfReader(str(pdf_file))
            page_count = len(reader.pages)

        except Exception as e:
            print(f"Gagal membaca {pdf_file.name}: {e}")
            page_count = 0

        total_pages += page_count

        print(
            f"{i}. {pdf_file.name} "
            f"({page_count} halaman)"
        )

        document_results.append(
            {
                "No": i,
                "Document Name": pdf_file.name,
                "Total Pages": page_count,
            }
        )

    print("-----------------------------------")
    print(f"Total Dokumen : {len(pdf_files)}")
    print(f"Total Halaman : {total_pages}")
    print("-----------------------------------\n")

    # ========================================================
    # LOAD PDF TANPA PREPROCESSING
    # ========================================================

    # Sengaja TIDAK menggunakan load_pdfs()
    # karena load_pdfs() pada pipeline utama sudah melakukan
    # preprocessing secara otomatis.

    loader = PyPDFDirectoryLoader(str(data_path))

    documents_raw = loader.load()

    if not documents_raw:
        raise ValueError(
            "Tidak ada dokumen yang berhasil diekstraksi."
        )

    # ========================================================
    # MEMBUAT VERSI PREPROCESSING
    # ========================================================

    documents_clean = deepcopy(documents_raw)

    for doc in documents_clean:
        doc.page_content = clean_text(
            doc.page_content
        )

    # ========================================================
    # SUMMARY RESULTS
    # ========================================================

    summary_results = []

    # ========================================================
    # MEMBUAT EXCEL
    # ========================================================

    output_path = "hasil_chunking.xlsx"

    with pd.ExcelWriter(output_path) as writer:

        # ====================================================
        # SHEET DOCUMENTS
        # ====================================================

        df_documents = pd.DataFrame(
            document_results
        )

        # Tambahkan total di bagian bawah
        df_documents.loc[len(df_documents)] = {
            "No": "",
            "Document Name": "TOTAL",
            "Total Pages": total_pages,
        }

        df_documents.to_excel(
            writer,
            sheet_name="Documents",
            index=False,
        )

        # ====================================================
        # LOOP PARAMETER CHUNKING
        # ====================================================

        for chunk_size, chunk_overlap in TEST_PARAMS:

            print(
                f"Testing chunk_size={chunk_size}, "
                f"overlap={chunk_overlap}"
            )

            # ================================================
            # CHUNK SEBELUM PREPROCESSING
            # ================================================

            chunks_raw = chunk_documents(
                documents_raw,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            # ================================================
            # CHUNK SESUDAH PREPROCESSING
            # ================================================

            chunks_clean = chunk_documents(
                documents_clean,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            # ================================================
            # HITUNG RATA-RATA PANJANG CHUNK
            # ================================================

            avg_chunk_length_raw = round(
                sum(
                    len(chunk.page_content)
                    for chunk in chunks_raw
                )
                / len(chunks_raw),
                2,
            )

            avg_chunk_length_clean = round(
                sum(
                    len(chunk.page_content)
                    for chunk in chunks_clean
                )
                / len(chunks_clean),
                2,
            )

            # ================================================
            # SIMPAN SUMMARY
            # ================================================

            summary_results.append(
                {
                    "Chunk Size": chunk_size,
                    "Chunk Overlap": chunk_overlap,
                    "Total Chunk (Before)": len(chunks_raw),
                    "Total Chunk (After)": len(chunks_clean),
                    "Average Length (Before)": (
                        avg_chunk_length_raw
                    ),
                    "Average Length (After)": (
                        avg_chunk_length_clean
                    ),
                }
            )

            # ================================================
            # DETAIL PER CHUNK
            # ================================================

            chunk_results = []

            max_chunks = max(
                len(chunks_raw),
                len(chunks_clean),
            )

            for i in range(max_chunks):

                # BEFORE PREPROCESSING
                if i < len(chunks_raw):

                    raw_content = (
                        chunks_raw[i].page_content
                    )

                    raw_length = len(raw_content)

                else:
                    raw_content = ""
                    raw_length = 0

                # AFTER PREPROCESSING
                if i < len(chunks_clean):

                    clean_content = (
                        chunks_clean[i].page_content
                    )

                    clean_length = len(clean_content)

                else:
                    clean_content = ""
                    clean_length = 0

                chunk_results.append(
                    {
                        "Chunk Number": i + 1,

                        "Chunk Length Before":
                            raw_length,

                        "Chunk Length After":
                            clean_length,

                        "Content Before Preprocessing":
                            raw_content,

                        "Content After Preprocessing":
                            clean_content,
                    }
                )

            df_chunk = pd.DataFrame(
                chunk_results
            )

            # ================================================
            # SIMPAN SHEET PER PARAMETER
            # ================================================

            sheet_name = (
                f"{chunk_size}_{chunk_overlap}"
            )

            df_chunk.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )

        # ====================================================
        # SHEET SUMMARY
        # ====================================================

        df_summary = pd.DataFrame(
            summary_results
        )

        df_summary.to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )

    # ========================================================
    # SELESAI
    # ========================================================

    print(
        f"\nBerhasil membuat {output_path}"
    )

    print("\nIsi Excel:")

    print("- Documents")
    print("- Summary")

    for chunk_size, chunk_overlap in TEST_PARAMS:
        print(
            f"- {chunk_size}_{chunk_overlap}"
        )


if __name__ == "__main__":
    main()
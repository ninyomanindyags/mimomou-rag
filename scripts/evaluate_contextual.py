"""Evaluasi retrieval + generation (RAGAS)
untuk pipeline Baseline vs SCG_CONTEXTUAL_SHORT.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
from src.utils.helpers import (
    format_docs,
    load_checkpoint,
    append_checkpoint,
    load_config,
)
from src.prompts.prompt_templates import SYSTEM_PROMPT


chat_model = load_llm()
config = load_config()

GT_PATH = config["paths"]["ground_truth_anthropic"]

CHECKPOINT_BASELINE = config["paths"]["eval_checkpoint_baseline"]
CHECKPOINT_SCG_CONTEXTUAL = config["paths"]["eval_checkpoint_scg_contextual"]


EVAL_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "CONTEXT:\n{context}\n\n"
            "Pertanyaan: {question}",
        ),
    ]
)


def is_record_complete(rec):
    if rec is None:
        return False

    answer = rec.get("answer")
    context = rec.get("context")

    if answer is None:
        return False

    if not str(answer).strip():
        return False

    if not context:
        return False

    return True


def run_pipeline(question, db, mode, max_retry=3):
    docs = retrieve_docs(db, mode, question)

    contexts = [d.page_content for d in docs]
    context_text = format_docs(docs) if docs else ""

    prompt = EVAL_PROMPT_TEMPLATE.invoke(
        {
            "context": context_text,
            "question": question,
        }
    )

    answer = None

    for attempt in range(max_retry):
        try:
            answer = chat_model.invoke(prompt).content
            break
        except Exception as e:
            print(
                f"  retry {attempt + 1}/{max_retry} - {e}"
            )
            time.sleep(5)

    return answer, contexts


def main():
    df = pd.read_excel(GT_PATH)

    done_baseline = load_checkpoint(
        CHECKPOINT_BASELINE,
        key_field="no",
    )

    done_scg = load_checkpoint(
        CHECKPOINT_SCG_CONTEXTUAL,
        key_field="no",
    )

    db_baseline = load_vector_store("Baseline")
    db_scg = load_vector_store("SCG_CONTEXTUAL_SHORT")

    # ========================================================
    # BASELINE
    # ========================================================

    for _, row in df.iterrows():
        no = int(row["No"])

        if (
            no in done_baseline
            and is_record_complete(done_baseline[no])
        ):
            continue

        question = row["Question"]

        print(
            f"[Baseline {no}/{len(df)}] {question}"
        )

        answer, context = run_pipeline(
            question,
            db_baseline,
            "Baseline",
        )

        rec = {
            "no": no,
            "answer": answer,
            "context": context,
        }

        append_checkpoint(
            CHECKPOINT_BASELINE,
            rec,
        )

        done_baseline[no] = rec

    # ========================================================
    # SCG_CONTEXTUAL_SHORT
    # ========================================================

    for _, row in df.iterrows():
        no = int(row["No"])

        if (
            no in done_scg
            and is_record_complete(done_scg[no])
        ):
            continue

        question = row["Question"]

        print(
            f"[SCG_CONTEXTUAL_SHORT {no}/{len(df)}] "
            f"{question}"
        )

        answer, context = run_pipeline(
            question,
            db_scg,
            "SCG_CONTEXTUAL_SHORT",
        )

        rec = {
            "no": no,
            "answer": answer,
            "context": context,
        }

        append_checkpoint(
            CHECKPOINT_SCG_CONTEXTUAL,
            rec,
        )

        done_scg[no] = rec

    # ========================================================
    # RELOAD CHECKPOINT
    # ========================================================

    done_baseline = load_checkpoint(
        CHECKPOINT_BASELINE,
        key_field="no",
    )

    done_scg = load_checkpoint(
        CHECKPOINT_SCG_CONTEXTUAL,
        key_field="no",
    )

    failed_baseline = [
        no
        for no, rec in done_baseline.items()
        if not is_record_complete(rec)
    ]

    failed_scg = [
        no
        for no, rec in done_scg.items()
        if not is_record_complete(rec)
    ]

    # ========================================================
    # UPDATE GROUND TRUTH
    # ========================================================

    if "Answer_Baseline" not in df.columns:
        df["Answer_Baseline"] = ""

    if "Answer_SCG_Contextual" not in df.columns:
        df["Answer_SCG_Contextual"] = ""

    if "Retrieved_Context_Baseline" not in df.columns:
        df["Retrieved_Context_Baseline"] = ""

    if "Retrieved_Context_SCG_Contextual" not in df.columns:
        df["Retrieved_Context_SCG_Contextual"] = ""

    for no, rec in done_baseline.items():
        mask = df["No"] == no

        df.loc[mask, "Answer_Baseline"] = (
            rec.get("answer") or ""
        )

        df.loc[mask, "Retrieved_Context_Baseline"] = (
            "\n---\n".join(
                rec.get("context") or []
            )
        )

    for no, rec in done_scg.items():
        mask = df["No"] == no

        df.loc[mask, "Answer_SCG_Contextual"] = (
            rec.get("answer") or ""
        )

        df.loc[mask, "Retrieved_Context_SCG_Contextual"] = (
            "\n---\n".join(
                rec.get("context") or []
            )
        )

    df.to_excel(GT_PATH, index=False)

    print(
        f"\nSpreadsheet terupdate: {GT_PATH}"
    )

    if failed_baseline or failed_scg:
        print(
            "\nMasih ada baris gagal. "
            "RAGAS belum dijalankan."
        )
        return

    # ========================================================
    # RAGAS
    # ========================================================

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig

    ragas_llm = LangchainLLMWrapper(chat_model)
    ragas_embeddings = LangchainEmbeddingsWrapper(
        load_embedder()
    )

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    run_config = RunConfig(
        max_workers=3,
        timeout=180,
        max_retries=3,
    )

    def build_dataset(answer_col, context_col):
        return Dataset.from_dict(
            {
                "question": df["Question"].tolist(),

                "answer": [
                    str(a)
                    if pd.notna(a)
                    else ""
                    for a in df[answer_col].tolist()
                ],

                "contexts": [
                    str(c).split("\n---\n")
                    if pd.notna(c)
                    else []
                    for c in df[context_col].tolist()
                ],

                "ground_truth": df["Ground_Truth"].tolist(),
            }
        )

    # ========================================================
    # BASELINE
    # ========================================================

    print(
        "\nMenjalankan RAGAS untuk pipeline BASELINE..."
    )

    result_baseline = evaluate(
        build_dataset(
            "Answer_Baseline",
            "Retrieved_Context_Baseline",
        ),
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
    )

    # ========================================================
    # SCG_CONTEXTUAL_SHORT
    # ========================================================

    print(
        "\nMenjalankan RAGAS untuk pipeline "
        "SCG_CONTEXTUAL_SHORT..."
    )

    result_scg = evaluate(
        build_dataset(
            "Answer_SCG_Contextual",
            "Retrieved_Context_SCG_Contextual",
        ),
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
    )

    # ========================================================
    # SAVE RESULT
    # ========================================================

    df_base = result_baseline.to_pandas()
    df_scg = result_scg.to_pandas()

    df_base.to_excel(
        config["paths"]["ragas_baseline_deepseek"],
        index=False,
    )

    df_scg.to_excel(
        config["paths"]["ragas_scg_contextual_deepseek"],
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    summary = pd.DataFrame(
        {
            "Metric": metric_names,

            "Baseline": [
                df_base[m].mean()
                for m in metric_names
            ],

            "SCG_Contextual": [
                df_scg[m].mean()
                for m in metric_names
            ],
        }
    )

    summary[
        "Selisih (SCG_Contextual - Baseline)"
    ] = (
        summary["SCG_Contextual"]
        - summary["Baseline"]
    )

    summary.to_excel(
        config["paths"][
            "ragas_contextual_comparison_deepseek"
        ],
        index=False,
    )

    print(
        "\n=== RINGKASAN PERBANDINGAN ==="
    )

    print(
        summary.to_string(index=False)
    )

    print(
        "\nFile hasil:"
    )

    print(
        config["paths"]["ragas_baseline_deepseek"]
    )

    print(
        config["paths"]["ragas_scg_contextual_deepseek"]
    )

    print(
        config["paths"][
            "ragas_contextual_comparison_deepseek"
        ]
    )


if __name__ == "__main__":
    main()
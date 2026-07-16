"""Fungsi-fungsi bantu yang dipakai lintas modul (config, checkpoint, formatting)."""
import json
import os
from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Load config.yaml sekali saja (di-cache), dipakai oleh semua modul."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def load_checkpoint(checkpoint_path: str, key_field: str) -> dict:
    """
    Baca file checkpoint JSONL (dipakai baik oleh build_scg_db.py maupun
    evaluate_retrieval.py) jadi dict {key_field_value: record}.

    Kalau ada baris duplikat (hasil retry), baris yang dibaca paling akhir
    otomatis menimpa yang lama karena dict assignment.
    """
    done = {}
    if not os.path.exists(checkpoint_path):
        return done

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                done[record[key_field]] = record
            except json.JSONDecodeError:
                continue

    return done


def append_checkpoint(checkpoint_path: str, record: dict):
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

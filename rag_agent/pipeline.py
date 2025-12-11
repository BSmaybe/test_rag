"""Утилиты для построения и поиска по локальному RAG-индексу."""
from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence

import faiss
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer


@dataclass
class Config:
    model_name: str
    dimension: int
    chunk_size: int
    chunk_overlap: int
    index_type: str = "IndexFlatIP"

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "index_type": self.index_type,
        }

    @classmethod
    def from_file(cls, path: Path) -> "Config":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


def clean_text(text: str) -> str:
    """Очистка HTML и технических символов."""
    raw = html.unescape(text or "")
    stripped = BeautifulSoup(raw, "html.parser").get_text(separator=" ")
    normalized = re.sub(r"\s+", " ", stripped)
    return normalized.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    cleaned = clean_text(text)
    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(cleaned):
        chunk = cleaned[start : start + chunk_size]
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def load_tickets(input_path: Path) -> pd.DataFrame:
    if input_path.suffix.lower() == ".csv":
        return pd.read_csv(input_path)
    if input_path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(input_path)
    raise ValueError("Поддерживаются только CSV, XLS, XLSX")


def prepare_chunks(
    df: pd.DataFrame, chunk_size: int = 500, chunk_overlap: int = 50
) -> List[Dict]:
    required_columns = {"ID", "Описание", "Решение"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют обязательные колонки: {sorted(missing)}")

    records: List[Dict] = []
    for _, row in df.iterrows():
        ticket_id = str(row["ID"])
        description = row.get("Описание", "")
        resolution = row.get("Решение", "")
        merged = f"Описание: {description}\nРешение: {resolution}"
        text_chunks = chunk_text(merged, chunk_size=chunk_size, overlap=chunk_overlap)
        payload_common = {
            "ticket_id": ticket_id,
            "source_fields": {
                "date": row.get("Дата"),
                "status": row.get("Статус"),
                "type": row.get("Тип"),
            },
        }
        for idx, chunk in enumerate(text_chunks):
            record = {
                **payload_common,
                "chunk_id": idx,
                "text": chunk,
            }
            records.append(record)
    return records


@lru_cache(maxsize=2)
def load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def encode_texts(
    texts: Sequence[str], model_name: str, batch_size: int = 32
) -> np.ndarray:
    model = load_model(model_name)
    embeddings = model.encode(
        list(texts), batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True
    )
    return embeddings.astype("float32")


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_index(
    index: faiss.Index,
    metadata: List[Dict],
    config: Config,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "index.faiss"))
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in metadata:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_index(index_dir: Path) -> tuple[faiss.Index, List[Dict], Config]:
    index_path = index_dir / "index.faiss"
    metadata_path = index_dir / "metadata.jsonl"
    config_path = index_dir / "config.json"

    index = faiss.read_index(str(index_path))
    metadata: List[Dict] = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))
    config = Config.from_file(config_path)
    return index, metadata, config


def persist_pipeline(
    df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    batch_size: int = 32,
    records: List[Dict] | None = None,
) -> None:
    if records is None:
        records = prepare_chunks(df, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = [r["text"] for r in records]
    embeddings = encode_texts(texts, model_name=model_name, batch_size=batch_size)
    index = build_faiss_index(embeddings)
    config = Config(
        model_name=model_name,
        dimension=embeddings.shape[1],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    save_index(index, records, config, output_dir=output_dir)


def search(
    query: str,
    index_dir: Path,
    top_k: int = 5,
    batch_size: int = 32,
    model_name: str | None = None,
) -> List[Dict]:
    index, metadata, config = load_index(index_dir)
    model_to_use = model_name or config.model_name
    query_vec = encode_texts([query], model_name=model_to_use, batch_size=batch_size)
    faiss.normalize_L2(query_vec)
    distances, ids = index.search(query_vec, top_k)
    results: List[Dict] = []
    for score, idx in zip(distances[0], ids[0]):
        if idx == -1:
            continue
        meta = metadata[idx]
        results.append({
            "score": float(score),
            "ticket_id": meta["ticket_id"],
            "chunk_id": meta["chunk_id"],
            "text": meta["text"],
            "source_fields": meta.get("source_fields", {}),
        })
    return results

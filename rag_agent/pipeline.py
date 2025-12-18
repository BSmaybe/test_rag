"""Утилиты для построения и поиска по локальному RAG-индексу."""
from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import faiss
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from FlagEmbedding import FlagReranker


REQUIRED_COLUMNS = {"ID", "Описание", "Решение"}
CANONICAL_ALIASES = {
    "id": "ID",
    "номер запроса": "ID",
    "описание": "Описание",
    "решение": "Решение",
    "описание решения": "Решение",
    "дата": "Дата",
    "дата/время регистрации": "Дата",
    "дата регистрации": "Дата",
    "статус": "Статус",
    "текущий статус": "Статус",
    "системный статус": "Статус",
    "тип": "Тип",
    "вид запроса": "Тип",
}
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


@dataclass
class Config:
    model_name: str
    dimension: int
    chunk_size: int
    chunk_overlap: int
    index_type: str = "IndexFlatIP"
    device: str = "cpu"

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "index_type": self.index_type,
            "device": self.device,
        }

    @classmethod
    def from_file(cls, path: Path) -> "Config":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{**{"device": "cpu"}, **data})


def clean_text(text: str) -> str:
    """Очистка HTML и технических символов."""
    raw = html.unescape(text or "")
    stripped = BeautifulSoup(raw, "html.parser").get_text(separator=" ")
    normalized = re.sub(r"\s+", " ", stripped)
    return normalized.strip()


def chunk_text(
    text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[str]:
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


def describe_columns(df: pd.DataFrame) -> Tuple[List[str], set[str]]:
    """Возвращает список колонок и отсутствующие обязательные поля."""

    columns = [str(col).strip() for col in df.columns]
    missing = REQUIRED_COLUMNS - set(columns)
    return sorted(columns), missing


def report_columns(columns: List[str], missing: set[str], reporter=print) -> None:
    reporter(f"Найдены колонки: {', '.join(columns)}")
    if missing:
        reporter(
            f"⚠️ Отсутствуют обязательные колонки: {', '.join(sorted(missing))}"
        )


def load_tickets(input_path: Path) -> pd.DataFrame:
    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    elif input_path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(input_path)
    else:
        raise ValueError("Поддерживаются только CSV, XLS, XLSX")

    normalized = normalize_headers(df)
    validate_schema(normalized)
    return normalized


def _normalize_column_name(name: object) -> str:
    cleaned = str(name or "")
    cleaned = (
        cleaned.replace("\ufeff", "")
        .replace("\u00a0", " ")
        .replace("\u200b", "")
        .replace("\u202f", " ")
    )
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip().casefold()


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Сопоставляет нестандартные названия колонок с целевыми алиасами."""

    rename_map: Dict[str, str] = {}

    for original in df.columns:
        normalized = _normalize_column_name(original)
        target = CANONICAL_ALIASES.get(normalized)
        if target:
            if original != target:
                rename_map[original] = target

    if rename_map:
        df = df.rename(columns=rename_map)

    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def validate_schema(df: pd.DataFrame) -> None:
    columns, missing = describe_columns(df)
    report_columns(columns, missing)
    if missing:
        raise ValueError(
            f"Отсутствуют обязательные колонки: {sorted(missing)}. "
            f"Найдены колонки: {', '.join(columns)}"
        )


def _normalize_source_value(value: object) -> str | None:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except TypeError:
        # Некоторые типы (например, datetime) не поддерживают pd.isna напрямую
        pass

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return str(value)


def _sanitize_record(record: Dict) -> Dict:
    sanitized = dict(record)
    source_fields = sanitized.get("source_fields") or {}
    sanitized["source_fields"] = {
        "date": _normalize_source_value(source_fields.get("date")),
        "status": _normalize_source_value(source_fields.get("status")),
        "type": _normalize_source_value(source_fields.get("type")),
    }
    sanitized["ticket_id"] = str(sanitized.get("ticket_id"))
    sanitized["chunk_id"] = int(sanitized.get("chunk_id", 0))
    sanitized["text"] = str(sanitized.get("text", ""))
    return sanitized


def prepare_chunks(
    df: pd.DataFrame,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict]:
    validate_schema(df)
    df = df.copy()
    df["ID"] = df["ID"].astype("string").str.strip()

    invalid_ids = df["ID"].isna() | (df["ID"] == "")
    if invalid_ids.any():
        invalid_count = int(invalid_ids.sum())
        raise ValueError(
            "Найдены пустые ID в "
            f"{invalid_count} строках. Заполните колонку ID перед индексацией."
        )

    total_rows = len(df)
    unique_ids = df["ID"].nunique(dropna=True)
    sample_ids = df["ID"].head(5).tolist()
    print(
        "[INGEST] Диагностика перед чанкингом: "
        f"строк={total_rows}, уникальных ID={unique_ids}, первые ID={sample_ids}"
    )

    records: List[Dict] = []
    for _, row in df.iterrows():
        ticket_id = row["ID"]
        description = row.get("Описание", "")
        resolution = row.get("Решение", "")
        merged = f"Описание: {description}\nРешение: {resolution}"
        text_chunks = chunk_text(merged, chunk_size=chunk_size, overlap=chunk_overlap)
        payload_common = {
            "ticket_id": ticket_id,
            "source_fields": {
                "date": _normalize_source_value(row.get("Дата")),
                "status": _normalize_source_value(row.get("Статус")),
                "type": _normalize_source_value(row.get("Тип")),
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
def load_model(model_name: str, device: str = "cpu") -> SentenceTransformer:
    return SentenceTransformer(model_name, device=device)


def encode_texts(
    texts: Sequence[str],
    model_name: str,
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    model = load_model(model_name, device=device)
    embeddings = model.encode(
        list(texts), batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True
    )
    return embeddings.astype("float32")


@lru_cache(maxsize=1)
def load_reranker(model_name: str, device: str = "cpu") -> FlagReranker:
    use_fp16 = device != "cpu"
    return FlagReranker(model_name, use_fp16=use_fp16, device=device)


@lru_cache(maxsize=1)
def load_tokenizer(model_name: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name)


def rerank_results(
    query: str,
    candidates: Sequence[Dict],
    model_name: str,
    device: str = "cpu",
    pairs: Sequence[tuple[str, str]] | None = None,
) -> List[Dict]:
    if not candidates:
        return []

    if not (isinstance(query, str) and len(query.strip()) > 10):
        print(
            f"[RERANK DEBUG] Skip rerank: invalid query of length {len(str(query)) if query is not None else 'None'}"
        )
        return []

    try:
        tokenizer = load_tokenizer(model_name)
    except Exception as exc:  # noqa: BLE001
        print(f"[RERANK ERROR] Failed to load tokenizer: {exc}")
        return []

    query_text = query.strip()
    query_tokens = tokenizer.encode(query_text, add_special_tokens=True)
    if not query_tokens:
        print("[RERANK DEBUG] Skip rerank: empty query tokenization")
        return []

    valid_candidates: List[Dict] = []
    pairs_to_score: List[tuple[str, str]] = []

    for row in candidates:
        text = row.get("text")
        if not (isinstance(text, str) and len(text.strip()) > 10):
            print(
                f"[SKIP RERANK] Bad chunk: {row.get('ticket_id')} "
                f"len={len(text) if isinstance(text, str) else 'None'}"
            )
            continue
        valid_candidates.append(row)
        if pairs is not None:
            try:
                _, provided_text = pairs[len(pairs_to_score)]
            except (IndexError, ValueError):
                provided_text = None
            text_to_use = (
                provided_text
                if isinstance(provided_text, str) and len(provided_text.strip()) > 10
                else text
            )
            pairs_to_score.append((query, text_to_use))
        else:
            text_to_use = text
            pairs_to_score.append((query, text_to_use))

        chunk_tokens = tokenizer.encode(text_to_use.strip(), add_special_tokens=True)
        if not chunk_tokens:
            print(
                f"[SKIP] Bad tokenization: {row.get('ticket_id')} "
                f"q_len={len(query_tokens)} c_len={len(chunk_tokens)}"
            )
            valid_candidates.pop()
            pairs_to_score.pop()

    print(f"[RERANK DEBUG] Final pairs count: {len(pairs_to_score)}")

    if not pairs_to_score:
        print("[RERANK DEBUG] No valid (query, chunk) pairs for reranking")
        return []

    try:
        reranker = load_reranker(model_name, device=device)
    except Exception as exc:  # noqa: BLE001
        print(f"[RERANK ERROR] Failed to load reranker: {exc}")
        return []

    try:
        scores = reranker.compute_score(pairs_to_score, normalize=True)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[RERANK ERROR] Failed to compute scores for {len(pairs_to_score)} pairs: {exc}"
        )

        fallback_candidates: List[Dict] = []
        fallback_scores: List[float] = []

        for idx, (pair, row) in enumerate(zip(pairs_to_score, valid_candidates)):
            try:
                pair_score = reranker.compute_score([pair], normalize=True)[0]
            except Exception as inner_exc:  # noqa: BLE001
                print(
                    "[RERANK ERROR] Drop pair "
                    f"idx={idx} ticket={row.get('ticket_id')} reason: {inner_exc}"
                )
                continue

            fallback_candidates.append(row)
            fallback_scores.append(pair_score)

        if not fallback_candidates:
            print("[RERANK DEBUG] All candidate pairs failed to score individually")
            return []

        valid_candidates = fallback_candidates
        scores = fallback_scores

    reranked: List[Dict] = []
    for row, score in zip(valid_candidates, scores):
        reranked.append({**row, "score": float(score)})

    return sorted(reranked, key=lambda row: row["score"], reverse=True)


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_index(
    index: faiss.Index,
    metadata: List[Dict],
    config: Config,
    index_dir: Path,
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    with (index_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in metadata:
            sanitized = _sanitize_record(row)
            f.write(json.dumps(sanitized, ensure_ascii=False) + "\n")
    (index_dir / "config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_index(index_dir: Path) -> tuple[faiss.Index, List[Dict], Config]:
    index_path = index_dir / "index.faiss"
    metadata_path = index_dir / "metadata.jsonl"
    config_path = index_dir / "config.json"

    for path in (index_path, metadata_path, config_path):
        if not path.exists():
            raise FileNotFoundError(f"Не найден файл индекса: {path}")

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
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    batch_size: int = 32,
    device: str = "cpu",
    records: List[Dict] | None = None,
) -> None:
    if records is None:
        records = prepare_chunks(df, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = [r["text"] for r in records]
    embeddings = encode_texts(
        texts, model_name=model_name, batch_size=batch_size, device=device
    )
    index = build_faiss_index(embeddings)
    config = Config(
        model_name=model_name,
        dimension=embeddings.shape[1],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        device=device,
    )
    save_index(index, records, config, index_dir=output_dir)


def update_index(
    df: pd.DataFrame,
    index_dir: Path,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    batch_size: int = 32,
    device: str | None = None,
    model_name: str | None = None,
) -> Dict:
    index_path = index_dir / "index.faiss"
    metadata_path = index_dir / "metadata.jsonl"
    config_path = index_dir / "config.json"

    if not (index_path.exists() and metadata_path.exists() and config_path.exists()):
        if model_name is None:
            raise ValueError(
                "Для создания нового индекса укажите модель (model_name). "
                "Индекс не найден в указанной директории.",
            )

        chunk_size_to_use = chunk_size or DEFAULT_CHUNK_SIZE
        chunk_overlap_to_use = chunk_overlap or DEFAULT_CHUNK_OVERLAP
        device_to_use = device or "cpu"
        records = prepare_chunks(
            df, chunk_size=chunk_size_to_use, chunk_overlap=chunk_overlap_to_use
        )
        persist_pipeline(
            df,
            output_dir=index_dir,
            model_name=model_name,
            chunk_size=chunk_size_to_use,
            chunk_overlap=chunk_overlap_to_use,
            batch_size=batch_size,
            device=device_to_use,
            records=records,
        )

        return {
            "added_ids": sorted({r["ticket_id"] for r in records}),
            "skipped_ids": [],
            "added_chunks": len(records),
            "total_chunks": len(records),
            "total_tickets": len({r["ticket_id"] for r in records}),
        }

    index, metadata, config = load_index(index_dir)
    chunk_size_to_use = chunk_size or config.chunk_size
    chunk_overlap_to_use = chunk_overlap or config.chunk_overlap
    device_to_use = device or config.device
    model_to_use = model_name or config.model_name

    if model_to_use != config.model_name:
        raise ValueError(
            "Нельзя обновить индекс другой моделью. Используйте модель из config.json."
        )

    records = prepare_chunks(
        df, chunk_size=chunk_size_to_use, chunk_overlap=chunk_overlap_to_use
    )

    existing_ids = {meta.get("ticket_id") for meta in metadata}
    new_records = [r for r in records if r["ticket_id"] not in existing_ids]
    skipped_ids = sorted({r["ticket_id"] for r in records} & existing_ids)

    if not new_records:
        return {
            "added_ids": [],
            "skipped_ids": skipped_ids,
            "added_chunks": 0,
            "total_chunks": len(metadata),
            "total_tickets": len({meta.get("ticket_id") for meta in metadata}),
        }

    texts = [r["text"] for r in new_records]
    embeddings = encode_texts(
        texts,
        model_name=model_to_use,
        batch_size=batch_size,
        device=device_to_use,
    )

    if embeddings.shape[1] != index.d:
        raise ValueError(
            "Размерность эмбеддингов не совпадает с существующим индексом. "
            "Используйте ту же модель, что в config.json."
        )

    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    metadata.extend(new_records)
    save_index(index, metadata, config, index_dir=index_dir)

    return {
        "added_ids": sorted({r["ticket_id"] for r in new_records}),
        "skipped_ids": skipped_ids,
        "added_chunks": len(new_records),
        "total_chunks": len(metadata),
        "total_tickets": len({meta.get("ticket_id") for meta in metadata}),
    }


def describe_index(index_dir: Path) -> Dict:
    index, metadata, config = load_index(index_dir)
    return {
        "chunks": len(metadata),
        "tickets": len({row.get("ticket_id") for row in metadata}),
        "config": config,
        "dimension": index.d,
    }


def load_metadata(index_dir: Path) -> List[Dict]:
    metadata_path = index_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Не найден файл метаданных: {metadata_path}")

    metadata: List[Dict] = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))
    return metadata


def clear_index(index_dir: Path) -> Dict[str, List[str]]:
    """Удаляет файлы индекса в указанной директории."""

    targets = ["index.faiss", "metadata.jsonl", "config.json"]
    removed: List[str] = []
    missing: List[str] = []

    for name in targets:
        path = index_dir / name
        if path.exists():
            path.unlink()
            removed.append(name)
        else:
            missing.append(name)

    try:
        index_dir.rmdir()
    except OSError:
        # Директория может содержать дополнительные файлы
        pass

    return {"removed": removed, "missing": missing}


def search(
    query: str,
    index_dir: Path,
    top_k: int = 5,
    batch_size: int = 32,
    model_name: str | None = None,
    device: str = "cpu",
    rerank: bool = False,
    reranker_model: str = "BAAI/bge-reranker-base",
    rerank_candidates: int = 30,
) -> List[Dict]:
    index, metadata, config = load_index(index_dir)
    model_to_use = model_name or config.model_name
    device_to_use = device or config.device
    candidates_to_fetch = rerank_candidates if rerank else top_k
    query_vec = encode_texts(
        [query], model_name=model_to_use, batch_size=batch_size, device=device_to_use
    )
    faiss.normalize_L2(query_vec)
    distances, ids = index.search(query_vec, min(candidates_to_fetch, index.ntotal))
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
    if rerank and results:
        pairs = [(query, ch["text"]) for ch in results if ch.get("text")]
        print(len(results), results[:1])
        print(f"[RERANK DEBUG] Retrieved: {len(results)}")
        for ch in results:
            print(f" - {ch.get('ticket_id')} | {len(ch.get('text', ''))} chars")
        print(f"[RERANK DEBUG] Pairs for reranking: {len(pairs)}")
        results = rerank_results(
            query=query,
            candidates=[ch for ch in results if ch.get("text")],
            model_name=reranker_model,
            device=device_to_use,
            pairs=pairs,
        )
    return results[:top_k]

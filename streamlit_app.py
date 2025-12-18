"""Streamlit UI для обновления и мониторинга FAISS-индекса."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Optional

import pandas as pd
import streamlit as st

from rag_agent.generator import generate_answer
from rag_agent.pipeline import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    describe_columns,
    describe_index,
    load_metadata,
    normalize_headers,
    report_columns,
    search,
    update_index,
)

DEFAULT_INDEX_DIR = Path(os.getenv("DEFAULT_INDEX_DIR", "data/index"))
DEFAULT_UPLOAD_DIR = Path(os.getenv("DEFAULT_UPLOAD_DIR", "data/uploaded"))
DEFAULT_LOG_DIR = Path(os.getenv("DEFAULT_LOG_DIR", "data/logs"))
DEFAULT_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_LLM_PATH = os.getenv(
    "DEFAULT_LLM_PATH", "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(uploaded_file)
    elif suffix in {".xls", ".xlsx"}:
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Поддерживаются только CSV, XLS, XLSX")
    return normalize_headers(df)


def archive_upload(uploaded_file, upload_dir: Path) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    target = upload_dir / f"{timestamp}_{uploaded_file.name}"
    target.write_bytes(uploaded_file.getbuffer())
    return target


def log_update(log_dir: Path, payload: dict) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": datetime.utcnow().isoformat(), **payload}
    with (log_dir / "updates.log").open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_existing_ids(index_dir: Path) -> set[str]:
    metadata_path = index_dir / "metadata.jsonl"
    if not metadata_path.exists():
        return set()
    metadata = load_metadata(index_dir)
    return {str(row.get("ticket_id")) for row in metadata}


def render_columns(df: pd.DataFrame) -> set[str]:
    columns, missing = describe_columns(df)
    report_columns(columns, missing, reporter=st.write)
    if missing:
        st.warning(
            "Добавление не будет выполнено, пока не появятся обязательные колонки"
            f" ({', '.join(sorted(missing))})."
        )
    return missing


def render_index_state(index_dir: Path) -> Optional[dict]:
    try:
        stats = describe_index(index_dir)
    except FileNotFoundError:
        st.warning("Индекс не найден: загрузите файл и нажмите \"Обновить индекс\".")
        return None

    st.subheader("Текущее состояние индекса")
    col1, col2, col3 = st.columns(3)
    col1.metric("Тикетов", stats["tickets"])
    col2.metric("Чанков", stats["chunks"])
    col3.metric("Размерность", stats["dimension"])

    cfg = stats["config"]
    st.caption(
        f"Модель: {cfg.model_name}, чанки: {cfg.chunk_size}/{cfg.chunk_overlap}, "
        f"устройство: {cfg.device}"
    )
    return stats


def main() -> None:
    st.set_page_config(page_title="FAISS индекс тикетов", layout="wide")
    st.title("FAISS индекс тикетов: загрузка и обновление")

    index_dir = Path(st.text_input("Путь к индексу", str(DEFAULT_INDEX_DIR)))
    stats = render_index_state(index_dir)

    st.subheader("Загрузка файла")
    uploaded_file = st.file_uploader("CSV или Excel", type=["csv", "xls", "xlsx"])
    chunk_size = st.number_input(
        "Размер чанка (символы)",
        min_value=50,
        max_value=2000,
        value=stats["config"].chunk_size if stats else DEFAULT_CHUNK_SIZE,
    )
    chunk_overlap = st.number_input(
        "Перекрытие чанков",
        min_value=0,
        max_value=500,
        value=stats["config"].chunk_overlap if stats else DEFAULT_CHUNK_OVERLAP,
    )

    if uploaded_file:
        with st.status("Загружаем файл...", expanded=True) as status:
            try:
                df = read_uploaded_file(uploaded_file)
            except Exception as exc:  # noqa: BLE001
                status.update(label="Не удалось загрузить файл", state="error")
                st.error(str(exc))
                df = None
            else:
                status.update(label="Файл загружен", state="complete")
        missing = render_columns(df) if df is not None else set()
    else:
        df = None
        missing = set()

    existing_ids = get_existing_ids(index_dir)
    ids_in_file: set[str] = set()
    if df is not None and "ID" in df.columns:
        ids_in_file = {str(val) for val in df["ID"].dropna().astype(str)}

    if df is not None:
        new_ids = ids_in_file - existing_ids
        already_indexed = ids_in_file & existing_ids
        st.caption(
            f"В файле уникальных ID: {len(ids_in_file)} | новых: {len(new_ids)} | "
            f"уже в индексе: {len(already_indexed)}"
        )

    if st.button("Обновить индекс", disabled=df is None or bool(missing)):
        if df is None:
            st.error("Сначала загрузите файл")
        elif missing:
            st.error("Заполните обязательные колонки перед обновлением индекса")
        else:
            uploaded_path = archive_upload(uploaded_file, DEFAULT_UPLOAD_DIR)
            started_at = perf_counter()
            try:
                with st.spinner("Обновляем индекс..."):
                    summary = update_index(
                        df=df,
                        index_dir=index_dir,
                        chunk_size=int(chunk_size),
                        chunk_overlap=int(chunk_overlap),
                        model_name=stats["config"].model_name if stats else DEFAULT_MODEL,
                    )
            except Exception as exc:  # noqa: BLE001
                log_update(
                    DEFAULT_LOG_DIR,
                    {
                        "file": uploaded_path.name,
                        "index_dir": str(index_dir),
                        "duration_seconds": round(perf_counter() - started_at, 3),
                        "added": 0,
                        "skipped": 0,
                        "errors": [str(exc)],
                    },
                )
                st.error(f"Ошибка обновления индекса: {exc}")
            else:
                log_update(
                    DEFAULT_LOG_DIR,
                    {
                        "file": uploaded_path.name,
                        "index_dir": str(index_dir),
                        "duration_seconds": round(perf_counter() - started_at, 3),
                        "added": len(summary.get("added_ids", [])),
                        "skipped": len(summary.get("skipped_ids", [])),
                        "errors": [],
                        "added_ids": summary.get("added_ids", []),
                        "skipped_ids": summary.get("skipped_ids", []),
                    },
                )
                st.success(
                    f"Добавлено чанков: {summary['added_chunks']}. "
                    f"Всего тикетов: {summary['total_tickets']} | Всего чанков: {summary['total_chunks']}"
                )
                if summary.get("added_ids"):
                    st.write("Добавленные ID:", ", ".join(summary["added_ids"]))
                if summary.get("skipped_ids"):
                    st.write("Пропущенные ID:", ", ".join(summary["skipped_ids"]))

    st.info("Загруженные файлы сохраняются в data/uploaded/, логи — в data/logs/updates.log.")

    st.subheader("Поиск и генерация ответа")
    query_text = st.text_area(
        "Вопрос оператора", placeholder="Например: Ошибка авторизации в личном кабинете"
    )
    top_k = st.slider("Сколько чанков вернуть", min_value=1, max_value=10, value=5)
    use_rerank = st.checkbox(
        "Использовать reranker (например, BGE)", value=False, help="Сначала выбираются top-30 в FAISS, затем результаты ранжируются"
    )
    reranker_model = st.text_input(
        "Модель reranker",
        value="BAAI/bge-reranker-base",
        disabled=not use_rerank,
    )
    mode = st.radio("Режим", ["Поиск", "Поиск + генерация"], horizontal=True)
    llm_path = st.text_input(
        "GGUF модель для генерации (например, mistral-7b-instruct-v0.2.Q4_K_M.gguf)",
        value=str(DEFAULT_LLM_PATH),
        disabled=mode == "Поиск",
    )
    max_tokens = st.number_input(
        "Лимит токенов ответа",
        min_value=128,
        max_value=1024,
        value=512,
        help="Чтобы уложиться в 10 секунд на CPU, держите ответ короче 1024 токенов.",
        disabled=mode == "Поиск",
    )
    n_ctx = st.number_input(
        "Контекстное окно модели (n_ctx)",
        min_value=2048,
        max_value=8192,
        value=4096,
        step=512,
        help="Для mistral рекомендуется не меньше 4096, минимум 2048.",
        disabled=mode == "Поиск",
    )

    if st.button("Запустить поиск", use_container_width=True):
        if not query_text.strip():
            st.error("Введите вопрос оператора")
        elif stats is None:
            st.error("Сначала создайте или укажите индекс")
        else:
            with st.spinner("Ищем похожие чанки..."):
                results = search(
                    query=query_text,
                    index_dir=index_dir,
                    top_k=int(top_k),
                    model_name=stats["config"].model_name,
                    device=stats["config"].device,
                    rerank=use_rerank,
                    reranker_model=reranker_model,
                )

            if not results:
                st.info("Ничего не найдено")
                return

            st.write("Найденные чанки:")
            for row in results:
                st.code(
                    f"[ID={row['ticket_id']} #{row['chunk_id']}] score={row['score']:.4f}\n{row['text']}",
                    language="markdown",
                )

            if mode == "Поиск + генерация":
                try:
                    with st.spinner("Генерируем ответ (llama-cpp)..."):
                        answer = generate_answer(
                            question=query_text,
                            chunks=results,
                            model_path=llm_path,
                            max_tokens=int(max_tokens),
                            n_ctx=int(n_ctx),
                        )
                except ImportError as exc:  # pragma: no cover - UI сообщение
                    st.error(str(exc))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Ошибка генерации ответа: {exc}")
                else:
                    st.success("Сгенерированный ответ")
                    st.write(answer)


if __name__ == "__main__":
    main()

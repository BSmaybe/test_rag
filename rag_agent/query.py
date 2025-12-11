"""CLI для поиска похожих тикетов по FAISS-индексу."""
from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import search


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Поиск в RAG-индексе")
    parser.add_argument("--index", required=True, help="Папка с index.faiss, metadata.jsonl, config.json")
    parser.add_argument("--query", required=True, help="Текст запроса оператора")
    parser.add_argument("--top-k", type=int, default=5, help="Количество результатов")
    parser.add_argument("--batch-size", type=int, default=32, help="Размер батча при инференсе")
    parser.add_argument("--model", help="Переопределить модель эмбеддингов из config.json")
    return parser


def format_row(row: dict) -> str:
    parts = [
        f"[ID={row['ticket_id']} #{row['chunk_id']}]",
        f"score={row['score']:.4f}",
        row["text"],
    ]
    source = row.get("source_fields") or {}
    extras = [f"{key}={value}" for key, value in source.items() if value]
    if extras:
        parts.append(f"meta: {', '.join(extras)}")
    return " | ".join(parts)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    index_dir = Path(args.index)
    results = search(
        query=args.query,
        index_dir=index_dir,
        top_k=args.top_k,
        batch_size=args.batch_size,
        model_name=args.model,
    )

    if not results:
        print("Ничего не найдено")
        return

    print("Найденные чанки:")
    for row in results:
        print(format_row(row))


if __name__ == "__main__":
    main()

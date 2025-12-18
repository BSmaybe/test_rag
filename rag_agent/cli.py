"""Единый CLI для индексации и поиска."""
from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import persist_pipeline, load_tickets, search, update_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG-агент для Naumen")
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest", help="Индексировать тикеты")
    ingest_parser.add_argument("--input", required=True, help="Путь к CSV/XLSX")
    ingest_parser.add_argument("--output", required=True, help="Папка для индекса")
    ingest_parser.add_argument(
        "--model",
        default="intfloat/multilingual-e5-small",
        help="Модель эмбеддингов",
    )
    ingest_parser.add_argument("--chunk-size", type=int, default=500)
    ingest_parser.add_argument("--chunk-overlap", type=int, default=50)
    ingest_parser.add_argument("--batch-size", type=int, default=32)

    query_parser = subparsers.add_parser("query", help="Поиск похожих тикетов")
    query_parser.add_argument("--index", required=True, help="Папка с индексом")
    query_parser.add_argument("--query", required=True, help="Текст запроса")
    query_parser.add_argument("--top-k", type=int, default=5)
    query_parser.add_argument("--batch-size", type=int, default=32)
    query_parser.add_argument("--model", help="Переопределить модель из config.json")

    update_parser = subparsers.add_parser("update", help="Добавить новые тикеты в индекс")
    update_parser.add_argument("--input", required=True, help="Путь к CSV/XLSX")
    update_parser.add_argument("--index", required=True, help="Папка с индексом")
    update_parser.add_argument("--chunk-size", type=int, default=None)
    update_parser.add_argument("--chunk-overlap", type=int, default=None)
    update_parser.add_argument("--batch-size", type=int, default=32)
    update_parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu"],
        help="Устройство инференса",
    )
    update_parser.add_argument(
        "--model",
        default=None,
        help="Модель эмбеддингов (по умолчанию из config.json)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "ingest":
            df = load_tickets(Path(args.input))
            print(f"Загружено {len(df)} тикетов")
            persist_pipeline(
                df=df,
                output_dir=Path(args.output),
                model_name=args.model,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                batch_size=args.batch_size,
            )
            print(f"Индекс сохранён в {args.output}")
        elif args.command == "query":
            results = search(
                query=args.query,
                index_dir=Path(args.index),
                top_k=args.top_k,
                batch_size=args.batch_size,
                model_name=args.model,
            )
            if not results:
                print("Ничего не найдено")
                return
            for row in results:
                print(f"[ID={row['ticket_id']} #{row['chunk_id']}] score={row['score']:.4f} \n{row['text']}\n")
        elif args.command == "update":
            df = load_tickets(Path(args.input))
            summary = update_index(
                df=df,
                index_dir=Path(args.index),
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                batch_size=args.batch_size,
                device=args.device,
                model_name=args.model,
            )
            added_ids = summary.get("added_ids") or []
            skipped_ids = summary.get("skipped_ids") or []

            if added_ids:
                print(f"Добавлены тикеты: {', '.join(added_ids)}")
            if skipped_ids:
                print(f"Пропущены существующие тикеты: {', '.join(skipped_ids)}")
            print(
                f"Индекс обновлён: +{summary['added_chunks']} чанков, всего тикетов "
                f"{summary['total_tickets']}, всего чанков {summary['total_chunks']}"
            )
        else:
            parser.print_help()
    except ValueError as exc:
        parser.exit(status=1, message=f"Ошибка: {exc}\n")


if __name__ == "__main__":
    main()

"""CLI для добавления новых тикетов в существующий индекс."""
from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import load_tickets, update_index


DEFAULT_MODEL = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Добавление тикетов в существующий FAISS-индекс")
    parser.add_argument("--input", required=True, help="Путь к CSV/XLSX с тикетами")
    parser.add_argument("--index", required=True, help="Папка с существующим индексом")
    parser.add_argument("--chunk-size", type=int, default=None, help="Размер чанка в символах (по умолчанию из config.json)")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Перекрытие чанков (по умолчанию из config.json)")
    parser.add_argument("--batch-size", type=int, default=32, help="Размер батча при инференсе")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu"],
        help="Устройство для инференса эмбеддингов (по умолчанию cpu)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Модель эмбеддингов (по умолчанию из config.json существующего индекса)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    index_dir = Path(args.index)

    df = load_tickets(input_path)
    summary = update_index(
        df=df,
        index_dir=index_dir,
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
        f"Индекс обновлён: +{summary['added_chunks']} чанков, "
        f"всего тикетов {summary['total_tickets']}, всего чанков {summary['total_chunks']}"
    )


if __name__ == "__main__":
    main()

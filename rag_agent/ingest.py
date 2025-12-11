"""CLI для индексации тикетов Naumen в локальный FAISS."""
from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import load_tickets, persist_pipeline, prepare_chunks


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Индексация тикетов для RAG")
    parser.add_argument("--input", required=True, help="Путь к CSV/XLSX с тикетами")
    parser.add_argument("--output", required=True, help="Папка для сохранения индекса")
    parser.add_argument(
        "--model",
        default="intfloat/multilingual-e5-small",
        help="Модель эмбеддингов (по умолчанию intfloat/multilingual-e5-small)",
    )
    parser.add_argument("--chunk-size", type=int, default=500, help="Размер чанка в символах")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Перекрытие чанков")
    parser.add_argument("--batch-size", type=int, default=32, help="Размер батча при инференсе")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu"],
        help="Устройство для инференса эмбеддингов (по умолчанию cpu)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    df = load_tickets(input_path)
    records = prepare_chunks(df, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    print(f"Загружено {len(df)} тикетов, подготовлено {len(records)} чанков")

    persist_pipeline(
        df=df,
        output_dir=output_dir,
        model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        device=args.device,
        records=records,
    )
    print(f"Индекс сохранён в {output_dir}")


if __name__ == "__main__":
    main()

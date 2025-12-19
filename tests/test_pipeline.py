import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
import numpy as np

from rag_agent.pipeline import normalize_headers, prepare_chunks, update_index


class PrepareChunksTestCase(unittest.TestCase):
    def test_raises_value_error_on_blank_ids(self) -> None:
        df = pd.DataFrame(
            {
                "ID": ["   ", None, "123"],
                "Описание": ["a", "b", "c"],
                "Решение": ["d", "e", "f"],
            }
        )

        with self.assertRaisesRegex(
            ValueError, r"Найдены пустые ID в 2 строках"
        ):
            prepare_chunks(df)

    def test_trims_and_casts_ids(self) -> None:
        df = pd.DataFrame(
            {
                "ID": ["  123  ", 456],
                "Описание": ["Первое", "Второе"],
                "Решение": ["Решение 1", "Решение 2"],
            }
        )

        records = prepare_chunks(df, chunk_size=128, chunk_overlap=0)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["ticket_id"], "123")
        self.assertEqual(records[1]["ticket_id"], "456")
        self.assertTrue(records[0]["text"].startswith("Описание:"))

    def test_maps_naumen_headers_and_counts_unique_ids(self) -> None:
        df = pd.DataFrame(
            {
                "Номер запроса": [1, 2, 3],
                "Описание": ["a", "b", "c"],
                "Описание решения": ["r1", "r2", "r3"],
            }
        )

        normalized = normalize_headers(df)

        self.assertIn("ID", normalized.columns)
        self.assertIn("Решение", normalized.columns)
        self.assertEqual(normalized["ID"].nunique(dropna=True), len(df))

    def test_normalizes_weird_or_missing_header_values(self) -> None:
        df = pd.DataFrame([[1, "desc", "res"]], columns=[pd.NA, "Номер запроса", "Описание решения"])

        normalized = normalize_headers(df)

        self.assertIn("ID", normalized.columns)
        self.assertEqual(normalized["ID"].iloc[0], "desc")

    def test_cleans_hyperlink_formulas_and_apostrophes(self) -> None:
        df = pd.DataFrame(
            {
                "ID": ['=HYPERLINK("http://x"; "123")'],
                "Описание": ['=HYPERLINK("http://x"; "Описание 1")'],
                "Решение": ['\'=HYPERLINK("http://x"; "Решение 1")'],
            }
        )

        records = prepare_chunks(df, chunk_size=128, chunk_overlap=0)

        self.assertEqual(records[0]["ticket_id"], "123")
        self.assertIn("Описание: Описание 1", records[0]["text"])
        self.assertIn("Решение: Решение 1", records[0]["text"])


class UpdateIndexSerializationTestCase(unittest.TestCase):
    def test_converts_timestamp_metadata(self) -> None:
        df = pd.DataFrame(
            {
                "ID": ["1"],
                "Описание": ["Описание"],
                "Решение": ["Решение"],
                "Дата": [pd.Timestamp("2024-01-15 12:00:00")],
                "Статус": ["В работе"],
                "Тип": ["Инцидент"],
            }
        )

        with TemporaryDirectory() as tmpdir, patch(
            "rag_agent.pipeline.encode_texts",
            return_value=np.zeros((1, 3), dtype="float32"),
        ):
            update_index(df, Path(tmpdir), model_name="dummy", batch_size=1)

            metadata_path = Path(tmpdir) / "metadata.jsonl"
            metadata_row = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(
                metadata_row["source_fields"].get("date"),
                "2024-01-15T12:00:00",
            )


if __name__ == "__main__":
    unittest.main()

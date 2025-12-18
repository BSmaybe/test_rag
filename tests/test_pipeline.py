import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
import numpy as np

from rag_agent.pipeline import prepare_chunks, update_index


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

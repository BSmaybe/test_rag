import unittest

import pandas as pd

from rag_agent.pipeline import prepare_chunks


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


if __name__ == "__main__":
    unittest.main()

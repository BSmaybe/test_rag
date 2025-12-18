import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import pandas as pd

from rag_agent import cli


class CLITestCase(unittest.TestCase):
    def test_ingest_reports_value_error(self) -> None:
        df = pd.DataFrame({"ID": [""], "Описание": [""], "Решение": [""]})

        with (
            mock.patch.object(cli, "load_tickets", return_value=df),
            mock.patch.object(
                cli, "persist_pipeline", side_effect=ValueError("Найдены пустые ID в 1 строках")
            ),
            mock.patch.object(
                sys, "argv", ["prog", "ingest", "--input", "tickets.csv", "--output", "idx"]
            ),
        ):
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()

            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                with self.assertRaises(SystemExit) as exit_info:
                    cli.main()

        self.assertEqual(exit_info.exception.code, 1)
        self.assertIn("Найдены пустые ID", stderr_buffer.getvalue())


if __name__ == "__main__":
    unittest.main()

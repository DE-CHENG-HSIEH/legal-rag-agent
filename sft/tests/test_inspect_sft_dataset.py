"""SFT 資料檢查器的異常格式回歸測試。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/01_inspect_sft_dataset.py"
SPEC = importlib.util.spec_from_file_location("inspect_sft_dataset_under_test", SCRIPT)
inspect_sft = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inspect_sft)


class SftDatasetInspectorTests(unittest.TestCase):
    """Ensure malformed object shapes become reportable errors, not crashes."""

    def test_jsonl_row_must_be_an_object(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "dataset.jsonl"
            path.write_text(
                "null\n" + json.dumps({"messages": []}) + "\n",
                encoding="utf-8",
            )

            records, errors = inspect_sft.load_jsonl(path)

        self.assertEqual(len(records), 1)
        self.assertEqual(errors[0]["error_type"], "record_not_object")

    def test_assistant_content_must_be_a_json_object(self) -> None:
        parsed, error = inspect_sft.parse_assistant_json("[]", line_number=7)

        self.assertIsNone(parsed)
        self.assertEqual(error["error_type"], "assistant_content_not_object")
        self.assertEqual(error["line_number"], 7)


if __name__ == "__main__":
    unittest.main()

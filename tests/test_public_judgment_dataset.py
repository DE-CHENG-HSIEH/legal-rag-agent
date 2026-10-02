"""驗證公開判決建置流程不改寫原文或洩漏本機來源欄位。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.build_public_judgment_dataset import (
    PUBLIC_FIELDS,
    TEXT_FIELDS,
    build_public_record,
    exclude_withdrawn_records,
    load_excluded_case_ids,
    validate_public_records,
)


class PublicJudgmentDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_record = {
            "case_id": "PI-TEST-00001",
            "split": "test",
            "case_no": "114年度測字第1號",
            "judgment_date": "民國114年01月01日",
            "court": "臺灣臺北地方法院",
            "judge": "王小明",
            "sentence_type": "罰金",
            "sentence_value": "新臺幣3,000元",
            "judgment": "被告陳小華犯公然侮辱罪，處罰金。",
            "search_facts": "被告在公開場所辱罵告訴人。",
            "search_sentencing_factors": "被告坦承犯行。",
            "original_crime_facts": "被告陳小華於公開場所辱罵告訴人林小美。",
            "original_sentencing": "審酌被告陳小華坦承犯行。",
            "source_file": "private-source.csv",
            "source_row_number": 42,
        }

    def test_public_record_preserves_judicial_yuan_text_verbatim(self) -> None:
        public_record = build_public_record(self.source_record)

        for field in TEXT_FIELDS:
            self.assertEqual(public_record[field], self.source_record[field])

    def test_public_record_excludes_local_preparation_provenance(self) -> None:
        public_record = build_public_record(self.source_record)

        self.assertEqual(set(public_record), set(PUBLIC_FIELDS))
        self.assertNotIn("source_file", public_record)
        self.assertNotIn("source_row_number", public_record)

    def test_public_contract_rejects_duplicate_case_ids(self) -> None:
        public_record = build_public_record(self.source_record)

        with self.assertRaisesRegex(ValueError, "duplicate case_id"):
            validate_public_records([public_record, public_record])

    def test_exclusion_list_ignores_comments_and_blank_lines(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "withdrawn_case_ids.txt"
            path.write_text(
                "# Source withdrawal\n\nPI-TEST-00001\nPI-TEST-00002\n",
                encoding="utf-8",
            )

            self.assertEqual(
                load_excluded_case_ids(path),
                {"PI-TEST-00001", "PI-TEST-00002"},
            )

    def test_unknown_withdrawal_case_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown case_id"):
            exclude_withdrawn_records(
                [self.source_record],
                {"PI-TEST-NOT-FOUND"},
            )


if __name__ == "__main__":
    unittest.main()

"""驗證公開判決資料契約與原始犯罪事實、量刑段落的保留。"""

import json
import unittest

from app.rag.documents import load_judgment_documents


class JudgmentDocumentTests(unittest.TestCase):
    def test_public_runtime_dataset_is_complete(self) -> None:
        documents = load_judgment_documents()

        self.assertEqual(len(documents), 2154)
        self.assertEqual(
            len({document.metadata["case_id"] for document in documents}),
            2154,
        )
        self.assertEqual(
            {document.metadata["split"] for document in documents},
            {"train", "validation", "test"},
        )

    def test_each_document_keeps_original_judgment_paragraphs(self) -> None:
        documents = load_judgment_documents()

        for document in documents:
            with self.subTest(case_id=document.metadata["case_id"]):
                self.assertTrue(document.metadata["original_crime_facts"].strip())
                self.assertTrue(document.metadata["original_sentencing"].strip())
                self.assertIn("客觀犯罪事實摘要", document.page_content)
                self.assertIn("客觀量刑因素摘要", document.page_content)

    def test_public_runtime_dataset_excludes_private_provenance(self) -> None:
        from app.rag.documents import JUDGMENTS_JSONL_PATH

        with JUDGMENTS_JSONL_PATH.open(encoding="utf-8") as file:
            first_record = json.loads(next(line for line in file if line.strip()))

        self.assertNotIn("source_file", first_record)
        self.assertNotIn("source_row_number", first_record)


if __name__ == "__main__":
    unittest.main()

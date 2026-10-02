"""公開重要法律見解語料的載入與來源追溯契約。"""

import unittest

from app.rag.authority_documents import (
    AUTHORITIES_CSV_PATH,
    load_authority_documents,
)


class AuthorityDocumentTests(unittest.TestCase):
    def test_public_authority_corpus_is_complete_and_traceable(self) -> None:
        self.assertEqual(
            AUTHORITIES_CSV_PATH.name,
            "public_insult_authorities.csv",
        )

        documents = load_authority_documents()

        self.assertEqual(len(documents), 24)
        for document in documents:
            self.assertIn("判決意旨原文：", document.page_content)
            self.assertTrue(document.metadata["source_id"])
            self.assertTrue(document.metadata["citation"])
            self.assertTrue(document.metadata["source_url"].startswith("https://"))
            self.assertTrue(document.metadata["verification_status"])


if __name__ == "__main__":
    unittest.main()

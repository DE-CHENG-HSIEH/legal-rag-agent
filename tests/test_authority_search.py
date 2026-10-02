"""重要法律見解檢索的來源多樣化測試。"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from app.rag.authority_search import search_authorities
from app.tools.authority_search import search_public_insult_authorities


def authority(source_id: str, citation: str) -> Document:
    return Document(
        page_content=f"{citation} 的已核對見解段落",
        metadata={"source_id": source_id, "citation": citation},
    )


class AuthoritySearchTests(unittest.TestCase):
    def test_results_keep_only_the_highest_ranked_passage_per_citation(self) -> None:
        candidates = [
            (authority("A-1", "來源 A"), 0.1, 0.99),
            (authority("A-2", "來源 A"), 0.2, 0.95),
            (authority("B-1", "來源 B"), 0.3, 0.90),
            (authority("C-1", "來源 C"), 0.4, 0.85),
        ]
        vector_store = Mock()
        vector_store.similarity_search_with_score.return_value = [
            (document, vector_score)
            for document, vector_score, _rerank_score in candidates
        ]

        with (
            patch(
                "app.rag.authority_search.get_authority_vector_store",
                return_value=vector_store,
            ),
            patch(
                "app.rag.authority_search.rerank_judgments",
                return_value=candidates,
            ) as rerank,
        ):
            results = search_authorities("言論自由與名譽權", initial_k=4, top_n=3)

        self.assertEqual(
            [result[0].metadata["source_id"] for result in results],
            ["A-1", "B-1", "C-1"],
        )
        vector_store.similarity_search_with_score.assert_called_once_with(
            query="言論自由與名譽權",
            k=12,
        )
        rerank.assert_called_once_with(
            query="言論自由與名譽權",
            docs_with_scores=vector_store.similarity_search_with_score.return_value,
            top_n=12,
        )

    @patch("app.tools.authority_search.search_authorities")
    def test_tool_returns_verified_text_as_source_artifact(
        self,
        search_mock: Mock,
    ) -> None:
        search_mock.return_value = [
            (
                Document(
                    page_content="檢索內容",
                    metadata={
                        "source_type": "憲法法庭判決",
                        "citation": "113年憲判字第3號",
                        "date": "2024-04-26",
                        "original_location": "理由第55段",
                        "source_url": "https://cons.judicial.gov.tw/example",
                        "verification_status": "已核對官方原文",
                        "authority_text": "這是已核對的官方原文。",
                    },
                ),
                0.1,
                0.9,
            )
        ]
        runtime = SimpleNamespace(stream_writer=Mock())

        content, artifact = search_public_insult_authorities.func(
            query="公然侮辱與言論自由",
            runtime=runtime,
            requested_count=1,
        )

        self.assertIn("這是已核對的官方原文。", content)
        self.assertEqual(
            artifact["entries"][0]["sections"][0]["text"],
            "這是已核對的官方原文。",
        )


if __name__ == "__main__":
    unittest.main()

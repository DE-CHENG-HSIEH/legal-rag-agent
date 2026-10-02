"""驗證相似判決工具的筆數契約與原文輸出邊界。"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from app.tools.judgment_search import (
    DEFAULT_JUDGMENT_COUNT,
    normalize_source_text_layout,
    resolve_judgment_count,
    search_similar_judgments,
)


class JudgmentSearchTests(unittest.TestCase):
    def test_source_layout_joins_cjk_hard_wraps_only(self) -> None:
        source = "因不\n滿店員不願販售香菸，辱罵告訴人等\n語。" "\n\n第二段原文。"

        self.assertEqual(
            normalize_source_text_layout(source),
            "因不滿店員不願販售香菸，辱罵告訴人等語。\n\n第二段原文。",
        )

    def test_explicit_count_overrides_default(self) -> None:
        self.assertEqual(resolve_judgment_count(10), 10)

    def test_missing_count_uses_internal_default(self) -> None:
        self.assertEqual(
            resolve_judgment_count(None),
            DEFAULT_JUDGMENT_COUNT,
        )

    def test_count_must_be_within_supported_range(self) -> None:
        for invalid_count in (0, 11, True):
            with self.subTest(invalid_count=invalid_count):
                with self.assertRaises(ValueError):
                    resolve_judgment_count(invalid_count)

    def test_tool_exposes_requested_count_to_the_model(self) -> None:
        schema = search_similar_judgments.tool_call_schema.model_json_schema()

        self.assertIn(
            "requested_count",
            schema["properties"],
        )
        self.assertNotIn("runtime", schema["properties"])

    @patch(
        "app.tools.judgment_search.retrieve_similar_judgments",
        return_value=[],
    )
    def test_explicit_count_is_forwarded_to_retrieval(
        self,
        retrieve_mock: Mock,
    ) -> None:
        runtime = SimpleNamespace(
            context=SimpleNamespace(),
            stream_writer=Mock(),
        )

        search_similar_judgments.func(
            query="便利商店內公開辱罵店員",
            runtime=runtime,
            requested_count=10,
        )

        retrieve_mock.assert_called_once_with(
            query="便利商店內公開辱罵店員",
            initial_k=40,
            top_n=10,
        )

    @patch(
        "app.tools.judgment_search.retrieve_similar_judgments",
    )
    def test_tool_displays_original_paragraphs_not_search_text(
        self,
        retrieve_mock: Mock,
    ) -> None:
        retrieve_mock.return_value = [
            (
                Document(
                    page_content="SFT 檢索用摘要",
                    metadata={
                        "court": "臺灣士林地方法院",
                        "case_no": "114年度審簡字第368號",
                        "date": "民國114年04月25日",
                        "case_type": "妨害名譽（公然侮辱）",
                        "judge": "法官李冠宜",
                        "sentence_type": "罰金",
                        "sentence_value": "壹仟元",
                        "original_crime_facts": "這是不\n應斷開的犯罪事實原文。",
                        "original_sentencing": "這是量刑段落\n原文。",
                        "judgment": "這是主文原文。",
                    },
                ),
                0.12,
                0.98,
            )
        ]
        runtime = SimpleNamespace(
            context=SimpleNamespace(),
            stream_writer=Mock(),
        )

        result, artifact = search_similar_judgments.func(
            query="公開辱罵告訴人",
            runtime=runtime,
            requested_count=1,
        )

        self.assertIn("【犯罪事實原文】", result)
        self.assertIn("資料來源：司法院公開裁判書", result)
        self.assertIn("這是不應斷開的犯罪事實原文。", result)
        self.assertIn("【量刑段落原文】", result)
        self.assertIn("這是量刑段落原文。", result)
        self.assertNotIn("不\n應", result)
        self.assertNotIn("SFT 檢索用摘要", result)
        self.assertEqual(artifact["artifact_type"], "legal_sources_v1")
        self.assertEqual(
            artifact["entries"][0]["sections"][0]["text"],
            "這是不應斷開的犯罪事實原文。",
        )


if __name__ == "__main__":
    unittest.main()

"""法務部法規客戶端的純函式與 HTML 邊界測試。"""

import unittest
from unittest.mock import MagicMock, patch

import requests

from app.laws.moj_law_client import (
    build_article_url,
    fetch_criminal_law_article,
    normalize_article_no,
    normalize_law_name,
)
from app.tools.criminal_law_lookup import lookup_criminal_law_article


class MojLawClientTests(unittest.TestCase):
    def test_law_aliases_are_normalized_and_unknown_laws_are_rejected(self) -> None:
        self.assertEqual(normalize_law_name("刑法"), "中華民國刑法")
        self.assertEqual(normalize_law_name("刑訴"), "刑事訴訟法")

        with self.assertRaisesRegex(ValueError, "僅支援"):
            normalize_law_name("民法")

    def test_article_numbers_are_normalized_without_silently_guessing(self) -> None:
        cases = {
            "第309條": "309",
            "３０９": "309",
            "第15條之1": "15-1",
            "15-1": "15-1",
            "第309條第1項": "309",
        }
        for raw_value, expected in cases.items():
            with self.subTest(raw_value=raw_value):
                self.assertEqual(normalize_article_no(raw_value), expected)

        with self.assertRaisesRegex(ValueError, "無法辨識"):
            normalize_article_no("第三百零九條")

    def test_article_url_uses_official_pcode_and_normalized_number(self) -> None:
        self.assertEqual(
            build_article_url("C0000001", "309"),
            "https://law.moj.gov.tw/LawClass/LawSingle.aspx?pcode=C0000001&flno=309",
        )

    @patch("app.laws.moj_law_client.create_moj_session")
    def test_fetch_extracts_only_the_requested_official_article(
        self,
        create_session: MagicMock,
    ) -> None:
        response = MagicMock()
        response.text = """
        <html><body>
          <div>第 309 條</div>
          <p>公然侮辱人者，處拘役或九千元以下罰金。</p>
          <div>憲法法庭裁判（新制）</div>
          <p>這一段不是法條本文。</p>
        </body></html>
        """
        session = create_session.return_value
        session.get.return_value = response

        record = fetch_criminal_law_article("刑法", "第309條")

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["law_name"], "中華民國刑法")
        self.assertEqual(record["article_no"], "309")
        self.assertEqual(
            record["article_content"],
            "公然侮辱人者，處拘役或九千元以下罰金。",
        )
        response.raise_for_status.assert_called_once_with()
        session.close.assert_called_once_with()

    @patch(
        "app.tools.criminal_law_lookup.fetch_criminal_law_article",
        side_effect=requests.ConnectionError("temporary failure"),
    )
    def test_tool_propagates_network_errors_for_agent_retry(
        self,
        _fetch_article: MagicMock,
    ) -> None:
        runtime = MagicMock()

        with self.assertRaises(requests.ConnectionError):
            lookup_criminal_law_article.func(
                law_name="刑法",
                article_no="309",
                runtime=runtime,
            )

        runtime.stream_writer.assert_called_once()

    @patch(
        "app.tools.criminal_law_lookup.fetch_criminal_law_article",
        return_value={
            "law_name": "中華民國刑法",
            "article_no": "355",
            "article_content": "意圖損害他人，以詐術使本人交付財物者。",
            "source_url": (
                "https://law.moj.gov.tw/LawClass/"
                "LawSingle.aspx?pcode=C0000001&flno=355"
            ),
        },
    )
    def test_tool_output_excludes_internal_model_instructions(
        self,
        _fetch_article: MagicMock,
    ) -> None:
        output = lookup_criminal_law_article.func(
            law_name="刑法",
            article_no="355",
            runtime=MagicMock(),
        )

        self.assertIn("【法條原文】", output)
        self.assertIn("意圖損害他人", output)
        self.assertIn("【官方來源】", output)
        self.assertNotIn("不得由語言模型", output)
        self.assertNotIn("摘要、改寫、刪減或補充", output)


if __name__ == "__main__":
    unittest.main()

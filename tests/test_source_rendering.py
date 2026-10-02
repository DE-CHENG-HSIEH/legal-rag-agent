"""法律來源 artifact 的確定性顯示契約。"""

import unittest

from langchain_core.messages import HumanMessage, ToolMessage

from app.agent.source_rendering import (
    compose_final_answer,
    make_legal_source_artifact,
    render_legal_sources,
)


class SourceRenderingTests(unittest.TestCase):
    def test_source_text_is_rendered_from_artifact_not_model_answer(self) -> None:
        artifact = make_legal_source_artifact(
            "statutes",
            [
                {
                    "title": "刑事法條查詢結果",
                    "metadata": [
                        {"label": "法規名稱", "value": "中華民國刑法"},
                    ],
                    "sections": [
                        {"heading": "法條原文", "text": "公然侮辱人者，處拘役。"},
                    ],
                }
            ],
        )
        messages = [
            HumanMessage(content="查詢刑法第309條"),
            ToolMessage(
                content="供模型分析的工具文字",
                tool_call_id="call-1",
                name="lookup_criminal_law_article",
                artifact=artifact,
            ),
        ]
        result = {
            "messages": messages,
            "structured_response": {
                "answer_markdown": (
                    "## AI 分析\n分析內容\n\n" "## 法律來源原文\n模型自行改寫的錯誤內容"
                )
            },
        }

        answer = compose_final_answer(result)

        self.assertIn("## AI 分析", answer)
        self.assertIn("公然侮辱人者，處拘役。", answer)
        self.assertNotIn("模型自行改寫", answer)
        self.assertTrue(answer.endswith(render_legal_sources(messages)))

    def test_only_current_turn_artifacts_are_rendered(self) -> None:
        old_artifact = make_legal_source_artifact(
            "statutes",
            [{"title": "舊來源", "sections": [{"heading": "原文", "text": "舊文"}]}],
        )
        new_artifact = make_legal_source_artifact(
            "statutes",
            [{"title": "新來源", "sections": [{"heading": "原文", "text": "新文"}]}],
        )
        messages = [
            HumanMessage(content="第一輪"),
            ToolMessage(
                content="舊工具結果",
                tool_call_id="call-1",
                name="lookup_criminal_law_article",
                artifact=old_artifact,
            ),
            HumanMessage(content="第二輪"),
            ToolMessage(
                content="新工具結果",
                tool_call_id="call-2",
                name="lookup_criminal_law_article",
                artifact=new_artifact,
            ),
        ]

        rendered = render_legal_sources(messages)

        self.assertIn("新文", rendered)
        self.assertNotIn("舊文", rendered)


if __name__ == "__main__":
    unittest.main()

"""驗證 Agent 提示、工具限制、輸出正規化與 checkpoint 契約。"""

import asyncio
import unittest
from types import SimpleNamespace

from langchain_core.messages import HumanMessage, ToolMessage

from app.agent.legal_agent import CHECKPOINT_SERIALIZER
from app.agent.middleware import SingleUseToolMiddleware
from app.agent.legal_prompts import SYSTEM_PROMPT
from app.agent.schemas import (
    LegalAnalysisOutput,
    normalize_markdown_text,
)
from app.tools.authority_search import resolve_authority_count


class AgentSchemaTests(unittest.TestCase):
    def test_system_prompt_does_not_force_tools_for_non_legal_tasks(self) -> None:
        self.assertIn(
            "不需要檢索或工具",
            SYSTEM_PROMPT,
        )
        self.assertIn(
            "不得先提出執行計畫並要求使用者再次確認",
            SYSTEM_PROMPT,
        )

    def test_system_prompt_enforces_tool_and_source_contracts(self) -> None:
        self.assertIn(
            "在同一個使用者請求中各最多呼叫一次",
            SYSTEM_PROMPT,
        )
        self.assertIn(
            "不得因排版",
            SYSTEM_PROMPT,
        )
        self.assertIn(
            "## 法律來源原文",
            SYSTEM_PROMPT,
        )
        self.assertIn(
            "## AI 比較",
            SYSTEM_PROMPT,
        )
        self.assertIn(
            "## AI 分析",
            SYSTEM_PROMPT,
        )
        self.assertLess(
            SYSTEM_PROMPT.index("## AI 比較"),
            SYSTEM_PROMPT.index("## 法律來源原文"),
        )
        self.assertIn(
            "不得在原文之後再重複相同分析",
            SYSTEM_PROMPT,
        )
        self.assertIn(
            "必須先使用 `## AI 分析`",
            SYSTEM_PROMPT,
        )
        self.assertIn(
            "再使用 `## 法律來源原文`",
            SYSTEM_PROMPT,
        )
        self.assertIn(
            "傳入 requested_count=2",
            SYSTEM_PROMPT,
        )
        self.assertIn(
            "不得摘要、改寫、刪減或補充",
            SYSTEM_PROMPT,
        )
        self.assertLess(
            SYSTEM_PROMPT.index("`## 研究模型預測`"),
            SYSTEM_PROMPT.rindex("`## 法律來源原文`"),
        )

    def test_authority_count_defaults_and_bounds(self) -> None:
        self.assertEqual(resolve_authority_count(None), 3)
        self.assertEqual(resolve_authority_count(1), 1)
        self.assertEqual(resolve_authority_count(5), 5)

        for invalid_count in (True, 0, 6, 2.5):
            with self.subTest(invalid_count=invalid_count):
                with self.assertRaises(ValueError):
                    resolve_authority_count(invalid_count)

    def test_consumed_high_cost_tool_is_hidden_from_later_model_calls(self) -> None:
        synthesis_model = object()
        middleware = SingleUseToolMiddleware(
            {"search_similar_judgments", "predict_public_insult_sentence"},
            synthesis_model=synthesis_model,
        )
        request = SimpleNamespace(
            model=object(),
            messages=[
                ToolMessage(
                    content="prediction result",
                    tool_call_id="call-1",
                    name="predict_public_insult_sentence",
                )
            ],
            tools=[
                SimpleNamespace(name="search_similar_judgments"),
                SimpleNamespace(name="predict_public_insult_sentence"),
                SimpleNamespace(name="lookup_criminal_law_article"),
            ],
        )

        def override(**changes):
            return SimpleNamespace(
                model=changes.get("model", request.model),
                messages=changes.get("messages", request.messages),
                tools=changes.get("tools", request.tools),
            )

        request.override = override
        captured_tool_names = []

        def handler(updated_request):
            self.assertIs(updated_request.model, synthesis_model)
            captured_tool_names.extend(tool.name for tool in updated_request.tools)
            return "handled"

        result = middleware.wrap_model_call(request, handler)

        self.assertEqual(result, "handled")
        self.assertEqual(
            captured_tool_names,
            ["search_similar_judgments", "lookup_criminal_law_article"],
        )

    def test_async_model_call_enforces_the_same_single_use_contract(self) -> None:
        synthesis_model = object()
        middleware = SingleUseToolMiddleware(
            {"search_similar_judgments"},
            synthesis_model=synthesis_model,
        )
        request = SimpleNamespace(
            model=object(),
            messages=[
                ToolMessage(
                    content="retrieval result",
                    tool_call_id="call-1",
                    name="search_similar_judgments",
                )
            ],
            tools=[
                SimpleNamespace(name="search_similar_judgments"),
                SimpleNamespace(name="lookup_criminal_law_article"),
            ],
        )

        def override(**changes):
            return SimpleNamespace(
                model=changes.get("model", request.model),
                messages=changes.get("messages", request.messages),
                tools=changes.get("tools", request.tools),
            )

        request.override = override

        async def run_call():
            async def handler(updated_request):
                self.assertIs(updated_request.model, synthesis_model)
                self.assertEqual(
                    [tool.name for tool in updated_request.tools],
                    ["lookup_criminal_law_article"],
                )
                return "handled asynchronously"

            return await middleware.awrap_model_call(request, handler)

        self.assertEqual(
            asyncio.run(run_call()),
            "handled asynchronously",
        )

    def test_consumed_tools_are_available_again_on_the_next_user_turn(self) -> None:
        routing_model = object()
        synthesis_model = object()
        middleware = SingleUseToolMiddleware(
            {"search_similar_judgments"},
            synthesis_model=synthesis_model,
        )
        request = SimpleNamespace(
            model=routing_model,
            messages=[
                HumanMessage(content="請搜尋三筆相似判決"),
                ToolMessage(
                    content="retrieval result",
                    tool_call_id="call-1",
                    name="search_similar_judgments",
                ),
                HumanMessage(content="請再用新的事實搜尋一次"),
            ],
            tools=[
                SimpleNamespace(name="search_similar_judgments"),
                SimpleNamespace(name="lookup_criminal_law_article"),
            ],
        )

        def override(**changes):
            return SimpleNamespace(
                model=changes.get("model", request.model),
                messages=changes.get("messages", request.messages),
                tools=changes.get("tools", request.tools),
            )

        request.override = override

        def handler(updated_request):
            self.assertIs(updated_request.model, routing_model)
            self.assertEqual(
                [tool.name for tool in updated_request.tools],
                ["search_similar_judgments", "lookup_criminal_law_article"],
            )
            return "handled on next turn"

        self.assertEqual(
            middleware.wrap_model_call(request, handler),
            "handled on next turn",
        )

    def test_checkpoint_serializer_round_trips_allowed_schemas(self) -> None:
        output = LegalAnalysisOutput(
            answer_markdown="測試回答",
        )

        encoded = CHECKPOINT_SERIALIZER.dumps_typed(output)
        decoded = CHECKPOINT_SERIALIZER.loads_typed(encoded)

        self.assertEqual(decoded, output)

    def test_literal_newlines_are_converted_to_markdown_lines(self) -> None:
        output = LegalAnalysisOutput(
            answer_markdown=("分析結果\\n\\n1) 相似判決\\n- 判決一\\n- 判決二")
        )

        self.assertEqual(
            output.answer_markdown,
            "分析結果\n\n1. 相似判決\n- 判決一\n- 判決二",
        )

    def test_excessive_blank_lines_are_collapsed(self) -> None:
        self.assertEqual(
            normalize_markdown_text("第一段\n\n\n\n第二段"),
            "第一段\n\n第二段",
        )


if __name__ == "__main__":
    unittest.main()

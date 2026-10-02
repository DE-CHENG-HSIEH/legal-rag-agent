"""Streamlit 首頁的必要設定與初始互動狀態測試。"""

import os
from pathlib import Path
import unittest
from unittest.mock import patch

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from streamlit.testing.v1 import AppTest

from app.agent.source_rendering import make_legal_source_artifact

APP_PATH = Path(__file__).resolve().parents[1] / "ui/streamlit_app.py"
APP_ICON_PATH = APP_PATH.parent / "assets/legal-ai-mark.png"
APP_HERO_PATH = APP_PATH.parent / "assets/legal-ai-hero.png"


class StreamlitAppTests(unittest.TestCase):
    def setUp(self) -> None:
        st.cache_resource.clear()

    def tearDown(self) -> None:
        st.cache_resource.clear()

    def run_app(self, api_key: str) -> AppTest:
        with patch.dict(os.environ, {"OPENAI_API_KEY": api_key}, clear=False):
            return AppTest.from_file(str(APP_PATH)).run(timeout=30)

    def test_missing_openai_key_shows_setup_message_without_traceback(self) -> None:
        app = self.run_app("")

        self.assertFalse(app.exception)
        self.assertEqual(len(app.error), 1)
        self.assertIn("OPENAI_API_KEY", app.error[0].value)
        self.assertFalse(app.chat_input)

    def test_configured_app_renders_task_starters_and_chat_input(self) -> None:
        app = self.run_app("synthetic-test-key")

        self.assertFalse(app.exception)
        self.assertTrue(APP_ICON_PATH.is_file())
        self.assertTrue(APP_HERO_PATH.is_file())
        self.assertEqual(len(app.chat_input), 1)
        self.assertEqual(
            {button.label for button in app.button},
            {
                "搜尋相似判決",
                "查詢法條與見解",
                "預測刑種與刑度群組",
                "進行完整案件分析",
            },
        )

    def test_failed_turn_recovers_from_last_completed_conversation(self) -> None:
        calls = []
        completed = [HumanMessage(content="第一輪"), AIMessage(content="已完成")]

        class FakeAgent:
            async def astream(self, inputs, *, config, **kwargs):
                calls.append((inputs, config))
                if len(calls) == 2:
                    yield ("values", {"messages": [AIMessage(content="未完成")]})
                    raise RuntimeError("synthetic stream failure")
                yield (
                    "values",
                    {
                        "messages": completed,
                        "structured_response": {"answer_markdown": "已完成"},
                    },
                )

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic-test-key"}),
            patch("app.agent.legal_agent.get_legal_agent", return_value=FakeAgent()),
        ):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
            app.chat_input[0].set_value("第一輪").run(timeout=30)
            app.chat_input[0].set_value("失敗的第二輪").run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(len(app.error), 1)
            app.chat_input[0].set_value("第三輪").run(timeout=30)

        self.assertFalse(app.exception)
        first_id = calls[0][1]["configurable"]["thread_id"]
        self.assertEqual(first_id, calls[1][1]["configurable"]["thread_id"])
        self.assertNotEqual(first_id, calls[2][1]["configurable"]["thread_id"])
        self.assertEqual(calls[2][0]["messages"][:-1], completed)
        self.assertEqual(calls[2][0]["messages"][-1]["content"], "第三輪")

    def test_rerun_keeps_source_text_identical_to_initial_answer(self) -> None:
        original = r"合成來源含有字面符號\n，不能替換。"
        artifact = make_legal_source_artifact(
            "statutes",
            [{"title": "合成來源", "sections": [{"heading": "原文", "text": original}]}],
        )

        class FakeAgent:
            async def astream(self, *args, **kwargs):
                yield (
                    "values",
                    {
                        "messages": [
                            HumanMessage(content="測試"),
                            ToolMessage(content="來源", tool_call_id="1", artifact=artifact),
                        ],
                        "structured_response": {"answer_markdown": "## AI 分析\n合成分析"},
                    },
                )

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic-test-key"}),
            patch("app.agent.legal_agent.get_legal_agent", return_value=FakeAgent()),
        ):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
            app.chat_input[0].set_value("測試").run(timeout=30)
            initial = next(item.value for item in app.markdown if original in item.value)
            app.run(timeout=30)

        self.assertFalse(app.exception)
        self.assertIn(initial, [item.value for item in app.markdown])


if __name__ == "__main__":
    unittest.main()

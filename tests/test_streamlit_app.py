"""Streamlit 首頁的必要設定與初始互動狀態測試。"""

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "ui/streamlit_app.py"
APP_ICON_PATH = APP_PATH.parent / "assets/legal-ai-mark.png"
APP_HERO_PATH = APP_PATH.parent / "assets/legal-ai-hero.png"


class StreamlitAppTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

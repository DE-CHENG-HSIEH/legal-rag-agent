"""Agent 結構化回應與 UI 顯示前的文字正規化契約。"""

import re

from pydantic import BaseModel, Field, field_validator


def normalize_markdown_text(value: str) -> str:
    """將結構化輸出中的跳脫換行轉為可顯示的 Markdown。"""

    if not isinstance(value, str):
        value = str(value)

    text = value.replace("\\r\\n", "\n").replace("\\n", "\n")
    text = re.sub(r"(?m)^(\d+)\)\s+", r"\1. ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class LegalAnalysisOutput(BaseModel):
    """UI 唯一需要的 Agent 最終輸出契約。"""

    answer_markdown: str = Field(
        description=(
            "給使用者閱讀的 AI 分析，可以使用 Markdown；法律來源原文由應用程式"
            "根據工具 artifact 另行附加，不得在此重貼。"
        )
    )

    @field_validator("answer_markdown")
    @classmethod
    def normalize_answer_markdown(cls, value: str) -> str:
        """Normalize escaped layout artifacts before the response reaches the UI."""

        return normalize_markdown_text(value)

"""把可預期的工具失敗轉成可安全呈現、可讓 Agent 繼續推理的訊息。"""

from typing import Any

import requests
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import AgentState
from langchain_core.messages import HumanMessage, ToolMessage

from app.agent.context import LegalAgentContext
from app.inference.public_insult_predictor import (
    PublicInsultPredictionCancelled,
)


def _tool_name(tool) -> str | None:
    """Read a LangChain or provider tool name without mutating its schema."""

    if isinstance(tool, dict):
        if isinstance(tool.get("name"), str):
            return tool["name"]
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            return function["name"]
        return None
    return getattr(tool, "name", None)


class SingleUseToolMiddleware(AgentMiddleware[AgentState[Any], LegalAgentContext, Any]):
    """Remove selected tools from later model calls in the current user turn.

    Prompt instructions improve model behavior but cannot guarantee it. Hiding a
    consumed high-cost tool from subsequent model calls in the same user request
    makes the one-call contract deterministic and avoids a redundant model turn.
    A later user request starts with the complete tool set again.
    """

    def __init__(
        self,
        tool_names: set[str] | frozenset[str],
        *,
        synthesis_model=None,
    ) -> None:
        """Configure the tools that become unavailable after their first result."""

        super().__init__()
        self.tool_names = frozenset(tool_names)
        self.synthesis_model = synthesis_model

    def wrap_model_call(self, request, handler):
        """Remove consumed tools and upgrade the post-retrieval synthesis model."""

        return handler(self._prepare_model_request(request))

    async def awrap_model_call(self, request, handler):
        """Apply the same one-call contract to asynchronous Agent streams."""

        return await handler(self._prepare_model_request(request))

    def _prepare_model_request(self, request):
        """Build the model request shared by synchronous and asynchronous paths."""

        # Check only messages produced after the latest user request. LangGraph
        # supplies earlier ToolMessages as conversational memory; treating those as
        # current-run calls would incorrectly disable tools for all later turns.
        current_turn_messages = request.messages
        for index in range(len(request.messages) - 1, -1, -1):
            if isinstance(request.messages[index], HumanMessage):
                current_turn_messages = request.messages[index + 1 :]
                break

        consumed_tools = {
            message.name
            for message in current_turn_messages
            if isinstance(message, ToolMessage) and message.name in self.tool_names
        }
        if not consumed_tools:
            return request

        available_tools = [
            tool for tool in request.tools if _tool_name(tool) not in consumed_tools
        ]
        overrides = {"tools": available_tools}
        if self.synthesis_model is not None:
            # Routing is cheap; source-faithful synthesis benefits from a stronger
            # model once retrieved text and prediction results are available.
            overrides["model"] = self.synthesis_model
        return request.override(**overrides)


def on_tool_error(
    exc: Exception,
    request,
) -> str | None:
    """處理已知錯誤；未知例外回傳 ``None``，保留框架的原始失敗。"""

    tool_name = request.tool_call.get(
        "name",
        "unknown_tool",
    )

    if isinstance(
        exc,
        PublicInsultPredictionCancelled,
    ):
        return "TAIDE LoRA 量刑預測已依使用者要求停止。"

    if isinstance(
        exc,
        ValueError,
    ):
        if tool_name == "predict_public_insult_sentence":
            return (
                "`predict_public_insult_sentence` 的輸入或模型輸出不符合契約。"
                "請確認犯罪事實與量刑因素均已提供，且案件內容未超過模型輸入上限；"
                "法院、民國年份與法官姓名均為選填，不得自行猜測。"
            )
        return (
            f"`{tool_name}` 工具輸入資料有誤。"
            "請檢查查詢內容、法規名稱或條號是否正確。"
        )

    if isinstance(
        exc,
        RuntimeError,
    ):
        if tool_name == "predict_public_insult_sentence":
            return (
                "TAIDE LoRA 量刑模型目前無法載入或完成推論。"
                "請檢查 HF_TOKEN、TAIDE 基礎模型存取權、網路連線、"
                "推論套件與可用記憶體。"
            )
        return f"`{tool_name}` 工具目前缺少必要設定。" "請檢查環境變數或系統設定。"

    if isinstance(
        exc,
        requests.RequestException,
    ):
        return (
            f"`{tool_name}` 工具連線外部服務失敗。"
            "請稍後再試，或改用目前已取得的資料回答。"
        )

    return None

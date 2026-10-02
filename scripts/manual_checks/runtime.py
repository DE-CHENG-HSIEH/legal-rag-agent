"""Runtime factory for invoking context-aware LangChain tools by hand."""

from typing import Any

from langchain.tools import ToolRuntime

from app.agent.context import LegalAgentContext


def build_manual_tool_runtime() -> ToolRuntime[LegalAgentContext, dict[str, Any]]:
    """Provide the state LangChain normally injects when an Agent calls a tool."""

    return ToolRuntime[LegalAgentContext, dict[str, Any]](
        state={},
        context=LegalAgentContext(),
        config={},
        stream_writer=lambda message: print(f"進度：{message}"),
        tool_call_id="manual-check",
        store=None,
    )

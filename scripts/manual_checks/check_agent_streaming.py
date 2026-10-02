"""檢查 LangGraph 串流事件可被逐步消費並產生最終結構化回答。"""

import asyncio
from uuid import uuid4

from app.agent.context import (
    LegalAgentContext,
)
from app.agent.legal_agent import (
    get_legal_agent,
)


async def main() -> None:
    """Run the same asynchronous streaming path used by the Streamlit UI."""

    agent = get_legal_agent()
    inputs = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "請查詢中華民國刑法第 309 條現行法條原文，"
                    "並將工具進度事件輸出到終端機。"
                ),
            }
        ]
    }
    config = {"configurable": {"thread_id": str(uuid4())}}

    async for event in agent.astream(
        inputs,
        config=config,
        context=LegalAgentContext(),
        stream_mode=["custom", "updates"],
    ):
        print(event)


if __name__ == "__main__":
    asyncio.run(main())

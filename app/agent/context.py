"""單次 Agent 執行需要、但不應進入對話 checkpoint 的 runtime 狀態。"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LegalAgentContext:
    """不應寫入對話歷史的單次執行狀態。"""

    # 取消訊號走 runtime context，避免被序列化到 LangGraph checkpoint。
    cancellation_event: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )

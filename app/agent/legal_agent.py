"""組裝模型、法律工具、可靠性 middleware 與對話 checkpoint。"""

from functools import lru_cache
import os
from typing import Any, cast

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)
from langchain.agents.middleware.types import AgentState

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import (
    InMemorySaver,
)
from langgraph.checkpoint.serde.jsonplus import (
    JsonPlusSerializer,
)

from app.agent.context import (
    LegalAgentContext,
)
from app.agent.middleware import (
    SingleUseToolMiddleware,
    on_tool_error,
)
from app.agent.legal_prompts import SYSTEM_PROMPT
from app.agent.schemas import (
    LegalAnalysisOutput,
)
from app.tools.authority_search import (
    search_public_insult_authorities,
)
from app.tools.criminal_law_lookup import (
    lookup_criminal_law_article,
)
from app.tools.judgment_search import (
    search_similar_judgments,
)
from app.tools.public_insult_prediction import (
    predict_public_insult_sentence,
)


load_dotenv()


DEFAULT_ROUTING_MODEL = os.getenv("OPENAI_ROUTING_MODEL", "gpt-5.4-nano")
DEFAULT_SYNTHESIS_MODEL = os.getenv("OPENAI_SYNTHESIS_MODEL", "gpt-5.4-mini")
DEFAULT_FALLBACK_MODEL = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-5.4")


TOOLS = [
    search_similar_judgments,
    search_public_insult_authorities,
    lookup_criminal_law_article,
    predict_public_insult_sentence,
]

SINGLE_USE_TOOL_NAMES = frozenset(
    {
        "search_similar_judgments",
        "search_public_insult_authorities",
        "predict_public_insult_sentence",
    }
)


def _single_use_limit_middleware(
    tool_name: str,
) -> AgentMiddleware[AgentState[Any], LegalAgentContext, Any]:
    # LangChain's ToolCallLimitState extends AgentState, but the framework's
    # invariant middleware list type cannot express that valid substitution.
    return cast(
        AgentMiddleware[AgentState[Any], LegalAgentContext, Any],
        ToolCallLimitMiddleware[Any, LegalAgentContext](
            tool_name=tool_name,
            run_limit=1,
            exit_behavior="continue",
        ),
    )


# Checkpoint 只允許反序列化本專案的結構化輸出，避免把任意 Python 類別
# 納入對話狀態的信任邊界。
CHECKPOINT_SERIALIZER = JsonPlusSerializer(
    allowed_msgpack_modules=[
        ("app.agent.schemas", "LegalAnalysisOutput"),
    ],
)


@lru_cache(maxsize=1)
def get_legal_agent(
    model_name: str = DEFAULT_ROUTING_MODEL,
):
    """建立並快取 Agent graph，避免每次 Streamlit rerun 重建模型客戶端。"""

    primary_model = ChatOpenAI(
        model=model_name,
        max_retries=0,
    )

    fallback_model = ChatOpenAI(
        model=DEFAULT_FALLBACK_MODEL,
        max_retries=0,
    )

    synthesis_model = ChatOpenAI(
        model=DEFAULT_SYNTHESIS_MODEL,
        max_retries=0,
    )

    agent = create_agent(
        model=primary_model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            # 工具成功或失敗一次後即從後續模型請求移除，從能力層阻止重複嘗試。
            SingleUseToolMiddleware(
                SINGLE_USE_TOOL_NAMES,
                synthesis_model=synthesis_model,
            ),
            # 限制高成本工具單輪只執行一次，避免模型以近似查詢重複消耗資源。
            _single_use_limit_middleware("search_similar_judgments"),
            _single_use_limit_middleware("search_public_insult_authorities"),
            _single_use_limit_middleware("predict_public_insult_sentence"),
            # OpenAI 模型失敗時先重試，再切換到較高階的備援模型。
            ModelRetryMiddleware(
                max_retries=2,
                backoff_factor=2.0,
                initial_delay=1.0,
            ),
            ModelFallbackMiddleware(
                fallback_model,
            ),
            # 將可預期的基礎設施錯誤轉成模型可安全說明的訊息。
            ToolErrorMiddleware(
                on_error=on_tool_error,
            ),
            ToolRetryMiddleware(
                max_retries=2,
                tools=[
                    "lookup_criminal_law_article",
                ],
                on_failure="error",
                backoff_factor=2.0,
                initial_delay=1.0,
            ),
        ],
        context_schema=LegalAgentContext,
        checkpointer=InMemorySaver(
            serde=CHECKPOINT_SERIALIZER,
        ),
        response_format=LegalAnalysisOutput,
        name="legal_rag_agent",
    )

    return agent

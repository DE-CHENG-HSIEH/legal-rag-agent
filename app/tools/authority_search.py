"""讓 Agent 查詢並逐字回傳已核對的重要法律見解。"""

from typing import Annotated

from langchain.tools import (
    ToolRuntime,
    tool,
)

from app.agent.context import (
    LegalAgentContext,
)
from app.agent.source_rendering import make_legal_source_artifact
from app.rag.authority_search import (
    search_authorities,
)


DEFAULT_AUTHORITY_COUNT = 3
MAX_AUTHORITY_COUNT = 5


def resolve_authority_count(
    requested_count: int | None,
) -> int:
    """限制單次來源數量，兼顧法律原文完整度與模型上下文大小。"""

    if requested_count is None:
        return DEFAULT_AUTHORITY_COUNT

    if isinstance(requested_count, bool) or not isinstance(requested_count, int):
        raise ValueError("重要法律見解回傳筆數必須是整數。")

    if not 1 <= requested_count <= MAX_AUTHORITY_COUNT:
        raise ValueError("重要法律見解回傳筆數必須介於 1 到 5 筆。")

    return requested_count


@tool(response_format="content_and_artifact")
def search_public_insult_authorities(
    query: str,
    runtime: ToolRuntime[LegalAgentContext],
    requested_count: Annotated[
        int | None,
        "要求回傳的重要法律見解筆數，限 1 到 5；未指定時省略。",
    ] = None,
) -> tuple[str, dict | None]:
    """
    搜尋公然侮辱罪相關的重要判決、
    憲法裁判及重要法律見解。
    """

    runtime.stream_writer("正在搜尋公然侮辱重要法律見解；首次使用會先建立本機索引……")

    result_count = resolve_authority_count(requested_count)

    results = search_authorities(
        query=query,
        initial_k=max(12, result_count * 4),
        top_n=result_count,
    )

    if not results:
        return "目前沒有找到相關的重要法律見解。", None

    runtime.stream_writer("重要法律見解搜尋與重排序完成。")

    output = [
        (
            "【本次檢索設定】\n"
            f"要求回傳：{result_count} 筆\n"
            f"實際取得：{len(results)} 筆"
        )
    ]
    source_entries = []

    for index, (
        document,
        _vector_score,
        _rerank_score,
    ) in enumerate(
        results,
        start=1,
    ):
        metadata = document.metadata
        authority_text = str(
            metadata.get("authority_text", document.page_content)
        ).strip()

        result_text = f"""
【重要法律見解 {index}】

來源類型：
{metadata.get("source_type", "未提供")}

案號或解釋字號：
{metadata.get("citation", "未提供")}

日期：
{metadata.get("date", "未提供")}

【以下為資料庫所保存之重要法律見解原文，
不得摘要、改寫、刪減或自行補充】

{authority_text}

原文位置：
{metadata.get("original_location", "未提供")}

官方／原始來源：
{metadata.get("source_url", "未提供")}

核對狀態：
{metadata.get("verification_status", "未提供")}
""".strip()

        output.append(result_text)

        source_entries.append(
            {
                "title": f"重要法律見解 {index}",
                "metadata": [
                    {
                        "label": "來源類型",
                        "value": metadata.get("source_type", "未提供"),
                    },
                    {
                        "label": "案號或解釋字號",
                        "value": metadata.get("citation", "未提供"),
                    },
                    {"label": "日期", "value": metadata.get("date", "未提供")},
                    {
                        "label": "原文位置",
                        "value": metadata.get("original_location", "未提供"),
                    },
                    {
                        "label": "官方來源",
                        "value": metadata.get("source_url", "未提供"),
                        "format": "url",
                    },
                    {
                        "label": "核對狀態",
                        "value": metadata.get("verification_status", "未提供"),
                    },
                ],
                "sections": [
                    {"heading": "重要法律見解原文", "text": authority_text},
                ],
            }
        )

    return "\n\n".join(output), make_legal_source_artifact(
        "legal_authorities",
        source_entries,
    )

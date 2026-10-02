"""讓 Agent 即時取得法務部官方刑事法條原文。"""

from langchain.tools import (
    ToolRuntime,
    tool,
)

from app.agent.context import (
    LegalAgentContext,
)
from app.agent.source_rendering import make_legal_source_artifact
from app.laws.moj_law_client import (
    fetch_criminal_law_article,
)


@tool(response_format="content_and_artifact")
def lookup_criminal_law_article(
    law_name: str,
    article_no: str,
    runtime: ToolRuntime[LegalAgentContext],
) -> tuple[str, dict | None]:
    """
    即時查詢法務部全國法規資料庫中的臺灣核心刑事法條原文。
    """

    runtime.stream_writer(f"正在查詢 {law_name} 第 {article_no} 條……")

    try:
        record = fetch_criminal_law_article(
            law_name=law_name,
            article_no=article_no,
        )

    except ValueError as error:
        return str(error), None

    if record is None:
        return (
            "全國法規資料庫目前查無指定法條。\n"
            f"法規：{law_name}\n"
            f"條號：{article_no}",
            None,
        )

    runtime.stream_writer("刑事法條查詢完成。")

    content = f"""
【刑事法條查詢結果】

法規名稱：
{record["law_name"]}

條號：
第 {record["article_no"]} 條

【法條原文】
{record["article_content"]}

【官方來源】
{record["source_url"]}

資料來源：
法務部全國法規資料庫
""".strip()

    artifact = make_legal_source_artifact(
        "statutes",
        [
            {
                "title": "刑事法條查詢結果",
                "metadata": [
                    {"label": "法規名稱", "value": record["law_name"]},
                    {"label": "條號", "value": f"第 {record['article_no']} 條"},
                    {
                        "label": "官方來源",
                        "value": record["source_url"],
                        "format": "url",
                    },
                    {"label": "資料來源", "value": "法務部全國法規資料庫"},
                ],
                "sections": [
                    {"heading": "法條原文", "text": record["article_content"]},
                ],
            }
        ],
    )
    return content, artifact

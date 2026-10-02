"""相似判決工具：檢索摘要、回傳公開原文，並限制單次上下文大小。"""

import re
from typing import Annotated

from langchain.tools import (
    ToolRuntime,
    tool,
)

from app.agent.context import (
    LegalAgentContext,
)
from app.rag.search import (
    retrieve_similar_judgments,
)


DEFAULT_JUDGMENT_COUNT = 5
MAX_JUDGMENT_COUNT = 10

# 司法院公開文字有時會保留網頁或文件的硬換行。Markdown 會把這類單行
# 換行折疊成空格，因此「不\n滿」會誤顯示成「不 滿」。這裡只處理不應
# 插入半形空格的中日韓文字、全形標點與數字邊界；空白行仍保留為
# 段落。
_CJK_LINE_WRAP_PATTERN = re.compile(
    r"(?<=[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef0-9])"
    r"[ \t]*\n[ \t]*"
    r"(?=[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef0-9])"
)


def normalize_source_text_layout(value: object) -> str:
    """移除公開法律文字中的版面硬換行，但不摘要或改寫原文。"""

    if value is None:
        return "未提供"

    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return "未提供"

    return _CJK_LINE_WRAP_PATTERN.sub("", text)


def resolve_judgment_count(
    requested_count: int | None,
) -> int:
    """解析單次相似判決查詢的回傳筆數。"""

    if requested_count is None:
        return DEFAULT_JUDGMENT_COUNT

    if isinstance(requested_count, bool) or not isinstance(requested_count, int):
        raise ValueError("相似判決回傳筆數必須是整數。")

    if not 1 <= requested_count <= MAX_JUDGMENT_COUNT:
        raise ValueError("相似判決回傳筆數必須介於 1 到 10 筆。")

    return requested_count


@tool
def search_similar_judgments(
    query: str,
    runtime: ToolRuntime[LegalAgentContext],
    requested_count: Annotated[
        int | None,
        "使用者在提示中明確要求的回傳筆數，限 1 到 10；未指定時省略。",
    ] = None,
) -> str:
    """
    根據犯罪事實與量刑要素搜尋語意相似的公然侮辱判決。
    """

    runtime.stream_writer(
        "正在進行相似判決向量檢索；首次使用會先建立本機索引，可能需要數分鐘……"
    )

    result_count = resolve_judgment_count(requested_count)

    results = retrieve_similar_judgments(
        query=query,
        initial_k=max(20, result_count * 4),
        top_n=result_count,
    )

    if not results:
        return "目前沒有找到可供參考的相似判決。"

    runtime.stream_writer("相似判決檢索與重排序完成。")

    output = [
        (
            "【本次檢索設定】\n"
            "資料來源：司法院公開裁判書（本專案僅進行結構化，原文採公開版本）\n"
            f"要求回傳：{result_count} 筆\n"
            f"實際取得：{len(results)} 筆"
        )
    ]

    for index, (
        document,
        _vector_score,
        _rerank_score,
    ) in enumerate(
        results,
        start=1,
    ):
        metadata = document.metadata

        original_crime_facts = normalize_source_text_layout(
            metadata.get("original_crime_facts")
        )
        original_sentencing = normalize_source_text_layout(
            metadata.get("original_sentencing")
        )
        original_judgment = normalize_source_text_layout(metadata.get("judgment"))

        result_text = f"""
【相似判決 {index}】

裁判法院：
{metadata.get("court", "未提供")}

裁判字號：
{metadata.get("case_no", "未提供")}

裁判日期：
{metadata.get("date", "未提供")}

裁判案由：
{metadata.get("case_type", "未提供")}

承審法官：
{metadata.get("judge", "未提供")}

處刑種類：
{metadata.get("sentence_type", "未提供")}

處刑輕重：
{metadata.get("sentence_value", "未提供")}

【犯罪事實原文】
{original_crime_facts}

【量刑段落原文】
{original_sentencing}

【主文原文】
{original_judgment}
""".strip()

        output.append(result_text)

    return "\n\n".join(output)

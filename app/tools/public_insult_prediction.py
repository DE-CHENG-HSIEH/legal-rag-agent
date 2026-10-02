"""LangChain tool：使用已發布的 TAIDE LoRA adapter 進行量刑群組預測。"""

import json
from typing import Annotated

from langchain.tools import ToolRuntime, tool

from app.agent.context import LegalAgentContext
from app.inference.public_insult_predictor import (
    get_public_insult_predictor,
    select_input_variant,
)


@tool
def predict_public_insult_sentence(
    criminal_facts: Annotated[str, "客觀犯罪事實摘要，不得加入尚未確認的事實。"],
    sentencing_factors: Annotated[
        str, "客觀量刑因素摘要，例如犯後態度、和解、前科與生活狀況。"
    ],
    runtime: ToolRuntime[LegalAgentContext],
    court: Annotated[
        str | None,
        "選填。已知時提供法院完整名稱，例如臺灣臺北地方法院；未知時省略。",
    ] = None,
    judgment_year_roc: Annotated[
        int | None,
        "選填。已知時提供預計裁判年份的民國年整數，例如114；未知時省略。",
    ] = None,
    judge_name: Annotated[
        str | None,
        "選填。已知時提供承審法官姓名；尚未分案或未知時省略。",
    ] = None,
) -> str:
    """預測公然侮辱案件的刑種、刑度群組，並生成量刑理由。

    犯罪事實與量刑因素為必要欄位；法院、民國年份及法官姓名均為選填。
    工具會依資料完整度自動選擇 A、B 或 C 組 LoRA adapter。輸出是研究用途
    的群組預測，不是精確刑度或判決保證。
    """
    predictor = get_public_insult_predictor()
    variant = select_input_variant(
        court=court,
        judgment_year_roc=judgment_year_roc,
        judge_name=judge_name,
    )
    if not predictor.is_loaded:
        runtime.stream_writer(
            "首次使用量刑模型，正在載入 TAIDE 與 A／B／C LoRA adapters……"
        )
    runtime.stream_writer(
        f"已依資料完整度選擇 {variant} 組，正在執行刑種、刑度群組與量刑理由預測……"
    )
    result = predictor.predict(
        criminal_facts=criminal_facts,
        sentencing_factors=sentencing_factors,
        court=court,
        judgment_year_roc=judgment_year_roc,
        judge_name=judge_name,
        cancellation_event=(runtime.context.cancellation_event),
    )
    runtime.stream_writer(f"TAIDE LoRA {variant} 組量刑預測完成。")
    return json.dumps(result, ensure_ascii=False, indent=2)

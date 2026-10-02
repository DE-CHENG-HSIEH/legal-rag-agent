"""SFT 生成評估的資料契約與可續跑輸出工具。

模型載入留在命令列入口，這裡只處理純資料邏輯，讓單元測試不需要 torch、MPS 或網路。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from public_insult_sft.evaluation import schema_errors


@dataclass(frozen=True)
class InferenceCase:
    case_id: str
    input_messages: list[dict[str, str]]
    reference: dict[str, Any]
    metadata: dict[str, Any]


def prepare_inference_cases(records: Iterable[dict[str, Any]]) -> list[InferenceCase]:
    """將 SFT JSONL 轉為 prompt/reference 配對，並在載入模型前驗證完整契約。"""
    cases: list[InferenceCase] = []
    seen_case_ids: set[str] = set()
    for position, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Dataset record {position} must be a JSON object.")
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(
                f"Dataset record {position} must contain prompt messages and one assistant answer."
            )

        assistant = messages[-1]
        if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
            raise ValueError(
                f"Dataset record {position} must end with an assistant answer."
            )
        try:
            reference = json.loads(assistant.get("content", ""))
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError(
                f"Dataset record {position} assistant answer is not valid JSON: {error}"
            ) from error
        errors = schema_errors(reference if isinstance(reference, dict) else None)
        if errors:
            raise ValueError(
                f"Dataset record {position} reference schema errors: {errors}"
            )

        input_messages: list[dict[str, str]] = []
        for message_index, message in enumerate(messages[:-1], start=1):
            if not isinstance(message, dict):
                raise ValueError(
                    f"Dataset record {position} message {message_index} must be an object."
                )
            role = message.get("role")
            content = message.get("content")
            if (
                role not in {"system", "user"}
                or not isinstance(content, str)
                or not content.strip()
            ):
                raise ValueError(
                    f"Dataset record {position} message {message_index} must be a non-empty system/user message."
                )
            input_messages.append({"role": role, "content": content})
        if not any(message["role"] == "user" for message in input_messages):
            raise ValueError(f"Dataset record {position} has no user prompt.")

        metadata_value = record.get("metadata")
        metadata = metadata_value if isinstance(metadata_value, dict) else {}
        case_id_value = metadata.get("case_id")
        case_id = str(case_id_value).strip() if case_id_value is not None else ""
        if not case_id:
            raise ValueError(f"Dataset record {position} has no metadata.case_id.")
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate metadata.case_id in dataset: {case_id}")
        seen_case_ids.add(case_id)

        cases.append(InferenceCase(case_id, input_messages, reference, metadata.copy()))
    if not cases:
        raise ValueError("Dataset contains no inference cases.")
    return cases


def load_completed_case_ids(path: Path) -> set[str]:
    """讀取既有成功輸出，供 --resume 跳過；損壞或重複列直接報錯。"""
    if not path.exists():
        return set()

    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Cannot resume from invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Cannot resume from non-object JSON at {path}:{line_number}"
                )
            case_id = record.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError(
                    f"Cannot resume record without case_id at {path}:{line_number}"
                )
            if case_id in completed:
                raise ValueError(
                    f"Duplicate case_id in existing prediction file: {case_id}"
                )
            # 失敗列不視為完成，讓後續執行可以重新嘗試；目前入口採 fail-fast，通常不會寫入失敗列。
            if not record.get("generation_error"):
                completed.add(case_id)
    return completed


def make_prediction_record(
    case: InferenceCase,
    *,
    generated_text: str,
    split: str,
    mode: str,
    base_model: str,
    adapter_path: str | None,
    generation_seconds: float,
) -> dict[str, Any]:
    """建立自包含的預測列，使後續計分不必再次讀取或修改原始 test JSONL。"""
    return {
        "case_id": case.case_id,
        "split": split,
        "model_mode": mode,
        "base_model": base_model,
        "adapter_path": adapter_path,
        "input_messages": case.input_messages,
        "reference": case.reference,
        "generated_text": generated_text,
        "generation_error": None,
        "generation_seconds": generation_seconds,
        "metadata": case.metadata,
    }

"""在載入大型模型前檢查 SFT JSONL 的格式、分布與文字長度。"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / "data/sft/public_insult/sft_prediction_and_generation_all.jsonl"
)

DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "sft/outputs/logs/sft_dataset_inspection_report.json"
)


def resolve_project_path(path: Path) -> Path:
    """Resolve CLI paths consistently when the script runs outside the repo root."""

    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="檢查公然侮辱 SFT JSONL 資料集格式。")

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="SFT JSONL 資料集路徑。",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="檢查報告輸出路徑。",
    )

    parser.add_argument(
        "--preview-count",
        type=int,
        default=2,
        help="終端機顯示前幾筆樣本。",
    )

    return parser.parse_args()


def load_jsonl(
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            raw_line = line.rstrip("\n")

            if not raw_line.strip():
                continue

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                errors.append(
                    {
                        "line_number": line_number,
                        "error_type": "json_decode_error",
                        "error_message": str(error),
                        "raw_line_preview": raw_line[:300],
                    }
                )
                continue

            if not isinstance(record, dict):
                errors.append(
                    {
                        "line_number": line_number,
                        "error_type": "record_not_object",
                        "raw_line_preview": raw_line[:300],
                    }
                )
                continue

            record["_line_number"] = line_number
            records.append(record)

    return records, errors


def get_message_by_role(
    messages: list[dict[str, Any]],
    role: str,
) -> dict[str, Any] | None:
    for message in messages:
        if isinstance(message, dict) and message.get("role") == role:
            return message

    return None


def safe_strip(value) -> str:
    if value is None:
        return ""

    return str(value).strip()


def parse_assistant_json(
    assistant_content: str,
    line_number: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        parsed = json.loads(assistant_content)
    except json.JSONDecodeError as error:
        return None, {
            "line_number": line_number,
            "error_type": "assistant_content_json_decode_error",
            "error_message": str(error),
            "assistant_content_preview": assistant_content[:500],
        }
    if not isinstance(parsed, dict):
        return None, {
            "line_number": line_number,
            "error_type": "assistant_content_not_object",
            "assistant_content_preview": assistant_content[:500],
        }
    return parsed, None


def inspect_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    role_errors: list[dict[str, Any]] = []
    assistant_json_errors: list[dict[str, Any]] = []

    sentence_type_counter: Counter[str] = Counter()
    court_counter: Counter[str] = Counter()

    empty_user_count = 0
    empty_assistant_count = 0
    missing_prediction_count = 0
    missing_sentence_type_count = 0
    missing_sentence_amount_count = 0
    missing_sentencing_reason_count = 0

    user_length_values: list[int] = []
    assistant_length_values: list[int] = []

    long_user_records: list[dict] = []
    long_assistant_records: list[dict] = []

    for record in records:
        line_number = record.get(
            "_line_number",
            None,
        )

        messages = record.get("messages")

        if not isinstance(messages, list):
            role_errors.append(
                {
                    "line_number": line_number,
                    "error_type": "messages_not_list",
                }
            )
            continue

        system_message = get_message_by_role(
            messages,
            "system",
        )
        user_message = get_message_by_role(
            messages,
            "user",
        )
        assistant_message = get_message_by_role(
            messages,
            "assistant",
        )

        if system_message is None or user_message is None or assistant_message is None:
            role_errors.append(
                {
                    "line_number": line_number,
                    "error_type": "missing_required_role",
                    "has_system": system_message is not None,
                    "has_user": user_message is not None,
                    "has_assistant": assistant_message is not None,
                }
            )
            continue

        user_content = safe_strip(user_message.get("content"))

        assistant_content = safe_strip(assistant_message.get("content"))

        user_length_values.append(len(user_content))

        assistant_length_values.append(len(assistant_content))

        if len(user_content) >= 6000:
            long_user_records.append(
                {
                    "line_number": line_number,
                    "user_length": len(user_content),
                    "preview": user_content[:300],
                }
            )

        if len(assistant_content) >= 6000:
            long_assistant_records.append(
                {
                    "line_number": line_number,
                    "assistant_length": len(assistant_content),
                    "preview": assistant_content[:300],
                }
            )

        if not user_content:
            empty_user_count += 1

        if not assistant_content:
            empty_assistant_count += 1
            continue

        assistant_json, assistant_error = parse_assistant_json(
            assistant_content=assistant_content,
            line_number=line_number,
        )

        if assistant_error is not None:
            assistant_json_errors.append(assistant_error)
            continue
        assert assistant_json is not None

        prediction = assistant_json.get(
            "prediction",
        )

        if not isinstance(prediction, dict):
            missing_prediction_count += 1
            continue

        sentence_type = safe_strip(prediction.get("sentence_type"))

        if sentence_type:
            sentence_type_counter[sentence_type] += 1
        else:
            missing_sentence_type_count += 1

        sentence_amount_numeric = prediction.get("sentence_amount_numeric")

        sentence_amount_raw = safe_strip(prediction.get("sentence_amount_raw"))

        sentence_amount_band = safe_strip(prediction.get("sentence_amount_band"))

        if (
            sentence_amount_numeric in [None, ""]
            and not sentence_amount_raw
            and not sentence_amount_band
        ):
            missing_sentence_amount_count += 1

        sentencing_reason = safe_strip(assistant_json.get("sentencing_reason"))

        if not sentencing_reason:
            missing_sentencing_reason_count += 1

        metadata = record.get(
            "metadata",
            {},
        )

        if isinstance(metadata, dict):
            court = safe_strip(metadata.get("court"))

            if court:
                court_counter[court] += 1

    valid_user_lengths = user_length_values or [0]
    valid_assistant_lengths = assistant_length_values or [0]

    report = {
        "total_records": len(records),
        "format_errors": {
            "role_error_count": len(role_errors),
            "empty_user_count": empty_user_count,
            "empty_assistant_count": empty_assistant_count,
            "assistant_json_error_count": len(assistant_json_errors),
            "missing_prediction_count": missing_prediction_count,
            "missing_sentence_type_count": missing_sentence_type_count,
            "missing_sentence_amount_count": missing_sentence_amount_count,
            "missing_sentencing_reason_count": missing_sentencing_reason_count,
        },
        "sentence_type_distribution": dict(sentence_type_counter),
        "court_distribution": dict(court_counter),
        "length_statistics": {
            "user_content_min_chars": min(valid_user_lengths),
            "user_content_max_chars": max(valid_user_lengths),
            "assistant_content_min_chars": min(valid_assistant_lengths),
            "assistant_content_max_chars": max(valid_assistant_lengths),
            "long_user_record_count_over_6000_chars": len(long_user_records),
            "long_assistant_record_count_over_6000_chars": len(long_assistant_records),
        },
        "sample_errors": {
            "role_errors": role_errors[:20],
            "assistant_json_errors": assistant_json_errors[:20],
            "long_user_records": long_user_records[:20],
            "long_assistant_records": long_assistant_records[:20],
        },
    }

    return report


def print_report(
    report: dict,
) -> None:
    print("=" * 80)
    print("SFT JSONL 資料集檢查報告")
    print("=" * 80)

    print()
    print("基本統計")
    print("-" * 80)
    print(f"總筆數：{report['total_records']}")

    print()
    print("格式問題")
    print("-" * 80)

    for key, value in report["format_errors"].items():
        print(f"{key}：{value}")

    print()
    print("處刑種類分布")
    print("-" * 80)

    for label, count in report["sentence_type_distribution"].items():
        print(f"{label}：{count}")

    print()
    print("法院分布")
    print("-" * 80)

    if report["court_distribution"]:
        for court, count in report["court_distribution"].items():
            print(f"{court}：{count}")
    else:
        print("metadata 中沒有 court 欄位，略過。")

    print()
    print("文字長度")
    print("-" * 80)

    for key, value in report["length_statistics"].items():
        print(f"{key}：{value}")

    print()
    print("檢查結論")
    print("-" * 80)

    errors = report["format_errors"]

    serious_error_count = (
        errors["role_error_count"]
        + errors["empty_user_count"]
        + errors["empty_assistant_count"]
        + errors["assistant_json_error_count"]
        + errors["missing_prediction_count"]
        + errors["missing_sentence_type_count"]
        + errors["missing_sentence_amount_count"]
        + errors["missing_sentencing_reason_count"]
    )

    if serious_error_count == 0:
        print("通過：資料集主要格式沒有發現嚴重問題。")
    else:
        print(
            "注意：資料集仍有格式或欄位問題，" "請先查看 report JSON 的 sample_errors。"
        )


def print_preview(
    records: list[dict],
    preview_count: int,
) -> None:
    print()
    print("=" * 80)
    print(f"前 {preview_count} 筆資料預覽")
    print("=" * 80)

    for index, record in enumerate(
        records[:preview_count],
        start=1,
    ):
        print()
        print(f"--- Sample {index} ---")
        print(
            json.dumps(
                {key: value for key, value in record.items() if key != "_line_number"},
                ensure_ascii=False,
                indent=2,
            )[:3000]
        )


def save_report(
    report: dict,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    args = parse_args()
    input_path = resolve_project_path(args.input)
    report_path = resolve_project_path(args.report)

    if not input_path.exists():
        raise FileNotFoundError(f"找不到資料集：{input_path}")

    records, jsonl_errors = load_jsonl(input_path)

    report = inspect_records(records)

    report["input_path"] = str(input_path)

    report["jsonl_load_errors"] = jsonl_errors

    print_report(report)

    print_preview(
        records=records,
        preview_count=args.preview_count,
    )

    save_report(
        report=report,
        path=report_path,
    )

    print()
    print("=" * 80)
    print("報告已輸出")
    print("=" * 80)
    print(report_path)


if __name__ == "__main__":
    main()

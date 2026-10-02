"""建立公然侮辱 SFT v2 的 A／B／C 消融資料集。

這支程式不從網路猜測法官姓名。它以清理流程留下的 master CSV 為權威來源，
逐案用 case_id 對應後，再交叉核對案號、法院、裁判日期與刑種；任一欄不一致就
整批停止，不會輸出部分補值的資料。產物保留原始精確刑度於 metadata 供稽核，
assistant 監督目標則改為刑種內三段式區間。
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data/sft/public_insult"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data/sft/public_insult_ablation"
SPLITS = ("train", "validation", "test")
VARIANTS = {
    "a": "犯罪事實與客觀量刑因素",
    "b": "A 加上法院與裁判年份",
    "c": "B 加上承審法官",
}

# band id 是模型輸出的穩定機器值。中文範圍由 min/max/unit 表達，避免把標點或
# 中文數字寫法納入分類標籤，讓評估能以完全一致的 enum 比對。
AMOUNT_BANDS: tuple[dict[str, Any], ...] = (
    {
        "id": "fine_1000_3000",
        "sentence_type": "罰金",
        "minimum": 1000,
        "maximum": 3000,
        "unit": "新臺幣元",
    },
    {
        "id": "fine_4000_5000",
        "sentence_type": "罰金",
        "minimum": 4000,
        "maximum": 5000,
        "unit": "新臺幣元",
    },
    {
        "id": "fine_6000_plus",
        "sentence_type": "罰金",
        "minimum": 6000,
        "maximum": None,
        "unit": "新臺幣元",
    },
    {
        "id": "detention_1_15",
        "sentence_type": "拘役",
        "minimum": 1,
        "maximum": 15,
        "unit": "日",
    },
    {
        "id": "detention_16_20",
        "sentence_type": "拘役",
        "minimum": 16,
        "maximum": 20,
        "unit": "日",
    },
    {
        "id": "detention_21_plus",
        "sentence_type": "拘役",
        "minimum": 21,
        "maximum": None,
        "unit": "日",
    },
)

SYSTEM_PROMPT = """你是臺灣刑事公然侮辱案件之判決結果預測與量刑理由生成模型。請依輸入之客觀犯罪事實與客觀量刑因素，輸出合法JSON，不得宣稱結果代表法院必然判決。

prediction必須包含sentence_type、sentence_amount_band、sentence_amount_min、sentence_amount_max與sentence_amount_unit。sentence_type只能是「拘役」或「罰金」。sentence_amount_band只能是fine_1000_3000、fine_4000_5000、fine_6000_plus、detention_1_15、detention_16_20、detention_21_plus之一，且必須與刑種、上下限及單位一致；無上限區間的sentence_amount_max使用null。最外層另須包含非空白的sentencing_reason。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare enriched A/B/C SFT datasets with sentencing bands."
    )
    parser.add_argument(
        "--master-csv",
        type=Path,
        required=True,
        help="Cleaned master CSV containing case_id and judge. This file is read-only.",
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the generated output root after every source record has passed validation.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_output_root(path: Path) -> Path:
    """限制產物只能寫在專案的 data/sft 子樹，避免 --overwrite 誤刪其他目錄。"""
    resolved = path.resolve()
    allowed_root = (PROJECT_ROOT / "data/sft").resolve()
    if resolved == allowed_root or allowed_root not in resolved.parents:
        raise ValueError(f"Output root must be a child of {allowed_root}: {resolved}")
    if path.is_symlink():
        raise ValueError(f"Output root cannot be a symbolic link: {path}")
    return resolved


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


def load_master(path: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {
            "case_id",
            "case_no",
            "court",
            "judgment_date",
            "sentence_type",
            "judge",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Master CSV is missing columns: {', '.join(sorted(missing))}"
            )
        rows = list(reader)

    lookup: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    for row in rows:
        case_id = row["case_id"].strip()
        if not case_id:
            raise ValueError("Master CSV contains an empty case_id.")
        if case_id in lookup:
            duplicates.append(case_id)
        lookup[case_id] = row
    if duplicates:
        raise ValueError(
            f"Master CSV contains duplicate case_id values: {duplicates[:5]}"
        )
    return lookup, list(reader.fieldnames or [])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_judge_name(raw_value: str) -> str:
    value = "".join(raw_value.split())
    value = re.sub(r"^法官", "", value)
    if not value:
        raise ValueError("Judge name is empty after normalization.")
    return value


def roc_year(judgment_date: str) -> int:
    match = re.fullmatch(r"民國(\d{2,3})年\d{2}月\d{2}日", judgment_date.strip())
    if not match:
        raise ValueError(f"Unsupported judgment_date format: {judgment_date!r}")
    return int(match.group(1))


def amount_band(sentence_type: str, amount: Any, unit: str) -> dict[str, Any]:
    if (
        isinstance(amount, bool)
        or not isinstance(amount, (int, float))
        or not math.isfinite(float(amount))
    ):
        raise ValueError(f"Invalid sentence amount: {amount!r}")
    numeric = float(amount)
    for definition in AMOUNT_BANDS:
        maximum = definition["maximum"]
        if (
            definition["sentence_type"] == sentence_type
            and definition["unit"] == unit
            and numeric >= definition["minimum"]
            and (maximum is None or numeric <= maximum)
        ):
            return definition
    raise ValueError(f"No sentencing band for {sentence_type=} {amount=} {unit=}")


def validated_enrichment(
    record: dict[str, Any],
    master_lookup: dict[str, dict[str, str]],
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    metadata = record.get("metadata")
    messages = record.get("messages")
    if not isinstance(metadata, dict) or not isinstance(messages, list):
        raise ValueError("Every record must contain metadata and chat messages.")
    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    if roles != ["system", "user", "assistant"] or len(roles) != len(messages):
        raise ValueError(
            "Every record must contain exactly system, user, assistant messages in that order."
        )
    case_id = str(metadata.get("case_id", "")).strip()
    master = master_lookup.get(case_id)
    if master is None:
        raise ValueError(f"No master row for case_id={case_id}")

    for field in ("case_no", "court", "judgment_date", "sentence_type"):
        source_value = str(metadata.get(field, "")).strip()
        master_value = str(master.get(field, "")).strip()
        if source_value != master_value:
            raise ValueError(
                f"Master mismatch for {case_id} field={field}: dataset={source_value!r}, master={master_value!r}"
            )

    try:
        answer = json.loads(messages[-1]["content"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid assistant JSON for case_id={case_id}") from error
    prediction = answer.get("prediction") if isinstance(answer, dict) else None
    if not isinstance(prediction, dict):
        raise ValueError(f"Missing prediction object for case_id={case_id}")
    reason = answer.get("sentencing_reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"Missing sentencing_reason for case_id={case_id}")

    sentence_type = str(prediction.get("sentence_type", "")).strip()
    unit = str(prediction.get("sentence_amount_unit", "")).strip()
    definition = amount_band(
        sentence_type, prediction.get("sentence_amount_numeric"), unit
    )
    return master, answer, definition


def prompt_for_variant(
    original: str,
    *,
    variant: str,
    court: str,
    year: int,
    judge_name: str,
) -> str:
    marker = "\n\n請輸出JSON"
    if marker not in original:
        raise ValueError(
            "User prompt does not contain the expected final JSON instruction marker."
        )
    body, _ = original.rsplit(marker, maxsplit=1)
    context_lines: list[str] = []
    if variant in {"b", "c"}:
        context_lines.extend(
            ("【案件制度資訊】", f"法院：{court}", f"裁判年份：民國{year}年")
        )
    if variant == "c":
        context_lines.append(f"承審法官：{judge_name}")
    context = "\n\n" + "\n".join(context_lines) if context_lines else ""
    return (
        body.rstrip()
        + context
        + "\n\n請輸出JSON，欄位包含prediction與sentencing_reason。"
    )


def transform_record(
    record: dict[str, Any],
    *,
    variant: str,
    master_lookup: dict[str, dict[str, str]],
    master_filename: str,
) -> dict[str, Any]:
    master, original_answer, definition = validated_enrichment(record, master_lookup)
    original_metadata = record["metadata"]
    original_prediction = original_answer["prediction"]
    judge_name = normalize_judge_name(master["judge"])
    year = roc_year(str(original_metadata["judgment_date"]))

    messages = [dict(message) for message in record["messages"]]
    messages[0]["content"] = SYSTEM_PROMPT
    messages[-2]["content"] = prompt_for_variant(
        messages[-2]["content"],
        variant=variant,
        court=str(original_metadata["court"]),
        year=year,
        judge_name=judge_name,
    )
    band_prediction = {
        "sentence_type": original_prediction["sentence_type"],
        "sentence_amount_band": definition["id"],
        "sentence_amount_min": definition["minimum"],
        "sentence_amount_max": definition["maximum"],
        "sentence_amount_unit": definition["unit"],
    }
    messages[-1]["content"] = json.dumps(
        {
            "prediction": band_prediction,
            "sentencing_reason": original_answer["sentencing_reason"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )

    metadata = dict(original_metadata)
    metadata.update(
        {
            "schema_version": "public_insult_sft_v2_band",
            "input_variant": variant.upper(),
            "judgment_year_roc": year,
            "judge_name": judge_name,
            "judge_source": master_filename,
            "judge_match_method": "case_id+case_no+court+judgment_date+sentence_type",
            "sentence_amount_band": definition["id"],
            "reference_sentence_amount_raw": original_prediction.get(
                "sentence_amount_raw"
            ),
            "reference_sentence_amount_numeric": original_prediction.get(
                "sentence_amount_numeric"
            ),
            "reference_sentence_amount_unit": original_prediction.get(
                "sentence_amount_unit"
            ),
        }
    )
    return {"messages": messages, "metadata": metadata}


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    temporary.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_partition(
    all_rows: list[dict[str, Any]], split_rows: dict[str, list[dict[str, Any]]]
) -> None:
    all_ids = [row["metadata"]["case_id"] for row in all_rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("The all dataset contains duplicate case_id values.")
    seen: set[str] = set()
    for split, rows in split_rows.items():
        ids = {row["metadata"]["case_id"] for row in rows}
        overlap = seen & ids
        if overlap:
            raise ValueError(
                f"Split {split} overlaps an earlier split: {sorted(overlap)[:5]}"
            )
        seen.update(ids)
    if seen != set(all_ids):
        raise ValueError(
            "train/validation/test do not form an exact partition of the all dataset."
        )


def main() -> None:
    args = parse_args()
    input_root = resolve_path(args.input_root)
    output_root = validate_output_root(resolve_path(args.output_root))
    master_path = resolve_path(args.master_csv)
    if not master_path.exists():
        raise FileNotFoundError(f"Master CSV not found: {master_path}")
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_root}. Use --overwrite to regenerate it."
        )

    all_rows = load_jsonl(input_root / "sft_prediction_and_generation_all.jsonl")
    split_rows = {split: load_jsonl(input_root / f"{split}.jsonl") for split in SPLITS}
    validate_partition(all_rows, split_rows)
    master_lookup, master_columns = load_master(master_path)

    # 先在記憶體轉換全部資料。任何補值或 schema 錯誤都會在碰觸既有輸出前終止。
    transformed: dict[str, dict[str, list[dict[str, Any]]]] = {}
    source_groups = {"all": all_rows, **split_rows}
    for variant in VARIANTS:
        transformed[variant] = {
            split: [
                transform_record(
                    row,
                    variant=variant,
                    master_lookup=master_lookup,
                    master_filename=master_path.name,
                )
                for row in rows
            ]
            for split, rows in source_groups.items()
        }

    if output_root.exists():
        shutil.rmtree(output_root)
    for variant, groups in transformed.items():
        variant_root = output_root / f"variant_{variant}"
        for split, rows in groups.items():
            atomic_write_jsonl(variant_root / f"{split}.jsonl", rows)

    manifest: dict[str, Any] = {
        "format_version": 1,
        "schema_version": "public_insult_sft_v2_band",
        "source": {
            "dataset_root": str(input_root),
            "master_csv_name": master_path.name,
            "master_csv_sha256": file_sha256(master_path),
            "master_columns": master_columns,
            "judge_match_method": "case_id+case_no+court+judgment_date+sentence_type",
        },
        "variants": VARIANTS,
        "amount_bands": list(AMOUNT_BANDS),
        "splits": {},
    }
    for split, rows in transformed["a"].items():
        manifest["splits"][split] = {
            "case_count": len(rows),
            "sentence_type_distribution": dict(
                sorted(
                    Counter(row["metadata"]["sentence_type"] for row in rows).items()
                )
            ),
            "amount_band_distribution": dict(
                sorted(
                    Counter(
                        row["metadata"]["sentence_amount_band"] for row in rows
                    ).items()
                )
            ),
            "judge_coverage_count": sum(
                bool(row["metadata"]["judge_name"]) for row in rows
            ),
        }
    atomic_write_json(output_root / "manifest.json", manifest)

    print(f"Prepared A/B/C datasets: {output_root}")
    print(f"Judge coverage: {len(all_rows)}/{len(all_rows)}")
    for split, summary in manifest["splits"].items():
        print(
            f"{split}: {summary['case_count']} cases; bands={summary['amount_band_distribution']}"
        )


if __name__ == "__main__":
    main()

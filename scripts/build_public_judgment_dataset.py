"""Build the version-controlled judgment corpus used by the Agent.

The runtime artifact preserves the Judicial Yuan's publicly released judgment
text verbatim. It removes only local preparation provenance that is unnecessary
for retrieval, such as spreadsheet filenames and row numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data/rag/public_insult_sft_retrieval.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data/public/public_insult_judgments.jsonl"
DEFAULT_EXCLUSIONS_PATH = PROJECT_ROOT / "data/public/withdrawn_case_ids.txt"

TEXT_FIELDS = (
    "judgment",
    "search_facts",
    "search_sentencing_factors",
    "original_crime_facts",
    "original_sentencing",
)
PUBLIC_FIELDS = (
    "case_id",
    "split",
    "case_no",
    "judgment_date",
    "court",
    "judge",
    "sentence_type",
    "sentence_value",
    *TEXT_FIELDS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the public judgment retrieval dataset."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--exclude-case-ids",
        type=Path,
        default=DEFAULT_EXCLUSIONS_PATH,
        help="Text file containing one withdrawn case_id per line.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"Dataset is empty: {path}")
    return records


def build_public_record(record: dict[str, Any]) -> dict[str, Any]:
    """Select runtime fields without changing any published judgment text."""

    missing = set(PUBLIC_FIELDS) - set(record)
    if missing:
        raise ValueError(
            f"Source record is missing fields: {', '.join(sorted(missing))}"
        )
    return {field: record[field] for field in PUBLIC_FIELDS}


def load_excluded_case_ids(path: Path) -> set[str]:
    """Load durable source-removal overrides used by every dataset rebuild."""

    if not path.is_file():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def exclude_withdrawn_records(
    records: list[dict[str, Any]],
    excluded_case_ids: set[str],
) -> list[dict[str, Any]]:
    """Apply withdrawals while rejecting misspelled or obsolete identifiers."""

    source_case_ids = {str(record.get("case_id", "")) for record in records}
    unknown_case_ids = excluded_case_ids - source_case_ids
    if unknown_case_ids:
        raise ValueError(
            "Withdrawal list contains unknown case_id values: "
            + ", ".join(sorted(unknown_case_ids))
        )
    return [
        record
        for record in records
        if str(record.get("case_id", "")) not in excluded_case_ids
    ]


def validate_public_records(records: list[dict[str, Any]]) -> None:
    case_ids = [str(record["case_id"]) for record in records]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Public dataset contains duplicate case_id values.")

    for index, record in enumerate(records, start=1):
        if set(record) != set(PUBLIC_FIELDS):
            raise ValueError(f"Unexpected field contract in public record {index}.")
        for field in TEXT_FIELDS:
            if not isinstance(record[field], str) or not record[field].strip():
                raise ValueError(f"Public record {index} has an empty {field} field.")


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    source_records = load_jsonl(args.input)
    excluded_case_ids = load_excluded_case_ids(args.exclude_case_ids)
    included_source_records = exclude_withdrawn_records(
        source_records,
        excluded_case_ids,
    )
    public_records = [build_public_record(record) for record in included_source_records]
    validate_public_records(public_records)
    write_jsonl(public_records, args.output)
    print(
        f"Built {len(public_records)} public judgment records "
        f"({len(excluded_case_ids)} excluded): {args.output}"
    )


if __name__ == "__main__":
    main()

"""把 SFT 案件集合與司法院公開判決原文對齊成私有準備資料。

此中間產物保留來源檔名與列號供稽核，不應直接發布；公開版本由
``build_public_judgment_dataset.py`` 選取必要欄位後產生。
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SFT_DIRECTORY = PROJECT_ROOT / "data/sft/public_insult_ablation/variant_c"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data/rag/public_insult_sft_retrieval.jsonl"

SOURCE_FACT_COLUMNS = (
    "原始犯罪事實",
    "犯罪事實",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "建立與 SFT 案件集合一致，並保留判決原始犯罪事實"
            "與量刑段落的 RAG 資料檔。"
        ),
    )
    parser.add_argument(
        "--master-csv",
        type=Path,
        required=True,
        help="SFT 清理後的 public_insult_cleaned_master_all.csv。",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="三份 Numbers 原始資料表匯出成 CSV 後所在的目錄。",
    )
    parser.add_argument(
        "--sft-dir",
        type=Path,
        default=DEFAULT_SFT_DIRECTORY,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    return parser.parse_args()


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_case_no(value) -> str:
    return re.sub(r"\s+", "", clean_text(value))


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def build_split_map(sft_directory: Path) -> dict[str, str]:
    split_by_case_id: dict[str, str] = {}

    for split in ("train", "validation", "test"):
        for record in load_jsonl(sft_directory / f"{split}.jsonl"):
            case_id = record["metadata"]["case_id"]
            if case_id in split_by_case_id:
                raise ValueError(f"重複的 SFT case_id：{case_id}")
            split_by_case_id[case_id] = split

    return split_by_case_id


def resolve_fact_column(source: pd.DataFrame) -> str:
    for column in SOURCE_FACT_COLUMNS:
        if column in source.columns:
            return column
    raise ValueError("來源資料表缺少原始犯罪事實欄位。")


def build_records(
    master_csv: Path,
    source_directory: Path,
    sft_directory: Path,
) -> list[dict]:
    master = pd.read_csv(master_csv).set_index("case_id", drop=False)
    sft_records = load_jsonl(sft_directory / "all.jsonl")
    split_by_case_id = build_split_map(sft_directory)

    source_files = {
        filename: pd.read_csv(source_directory / filename)
        for filename in master["source_file"].dropna().unique()
    }

    records: list[dict] = []
    seen_case_ids: set[str] = set()

    # case_id 是清理、SFT 與 RAG 三條 pipeline 的唯一 join key；案號再做
    # 第二層交叉核對，避免試算表列位移後把原文配到錯誤案件。
    for sft_record in sft_records:
        case_id = sft_record["metadata"]["case_id"]
        if case_id in seen_case_ids:
            raise ValueError(f"重複的 SFT case_id：{case_id}")
        seen_case_ids.add(case_id)

        if case_id not in master.index:
            raise ValueError(f"master CSV 找不到 case_id：{case_id}")
        if case_id not in split_by_case_id:
            raise ValueError(f"找不到 SFT split：{case_id}")

        master_row = master.loc[case_id]
        source_file = clean_text(master_row["source_file"])
        source = source_files[source_file]

        # Numbers 的列號包含標題列，pandas 的 iloc 由 0 開始。
        source_index = int(master_row["source_row_number"]) - 2
        if not 0 <= source_index < len(source):
            raise ValueError(f"來源列號超出範圍：{case_id} -> {source_index}")

        source_row = source.iloc[source_index]
        if normalize_case_no(source_row["裁判字號"]) != normalize_case_no(
            master_row["case_no"]
        ):
            raise ValueError(f"裁判字號無法對應：{case_id}")

        fact_column = resolve_fact_column(source)
        original_crime_facts = clean_text(source_row[fact_column])
        original_sentencing = clean_text(source_row["原始量刑"])

        if not original_crime_facts or not original_sentencing:
            raise ValueError(f"判決原文欄位為空：{case_id}")

        records.append(
            {
                "case_id": case_id,
                "split": split_by_case_id[case_id],
                "court": clean_text(master_row["court"]),
                "case_no": clean_text(master_row["case_no"]),
                "judgment_date": clean_text(master_row["judgment_date"]),
                "judge": clean_text(master_row["judge"]),
                "sentence_type": clean_text(master_row["sentence_type"]),
                "sentence_value": clean_text(source_row["處刑輕重"]),
                "judgment": clean_text(source_row["主文"]),
                "search_facts": clean_text(master_row["objective_facts_for_sft"]),
                "search_sentencing_factors": clean_text(
                    master_row["sentencing_factors_for_sft"]
                ),
                "original_crime_facts": original_crime_facts,
                "original_sentencing": original_sentencing,
                "source_file": source_file,
                "source_row_number": int(master_row["source_row_number"]),
            }
        )

    if len(records) != len(split_by_case_id):
        raise ValueError("all.jsonl 與 train/validation/test 合併後的案件數不一致。")

    return records


def write_jsonl(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(output_path)


def main() -> None:
    args = parse_args()
    records = build_records(
        master_csv=args.master_csv,
        source_directory=args.source_dir,
        sft_directory=args.sft_dir,
    )
    write_jsonl(records, args.output)
    print(f"已建立 {len(records)} 筆 SFT 同源判決檢索資料：{args.output}")


if __name__ == "__main__":
    main()

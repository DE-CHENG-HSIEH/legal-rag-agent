"""把公開判決語料轉成兼顧檢索品質與原文呈現的 LangChain 文件。"""

import json
from pathlib import Path

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[2]
JUDGMENTS_JSONL_PATH = PROJECT_ROOT / "data/public/public_insult_judgments.jsonl"


def clean_text(value) -> str:
    """
    將資料欄位轉成乾淨字串。
    """
    if value is None:
        return ""

    return str(value).strip()


def build_judgment_document(record: dict) -> Document:
    """將一筆判決轉換成供向量檢索與結果呈現使用的文件。"""

    facts = clean_text(record["search_facts"])
    sentencing = clean_text(record["search_sentencing_factors"])

    # 嵌入與重排序只使用已整理的客觀摘要；法院原始犯罪事實、量刑段落與
    # 主文留在 metadata，工具回傳結果時才展示，避免摘要取代法律來源原文。
    page_content = f"""
客觀犯罪事實摘要：
{facts}

客觀量刑因素摘要：
{sentencing}
""".strip()

    metadata = {
        "case_id": clean_text(record["case_id"]),
        "split": clean_text(record["split"]),
        "case_no": clean_text(record["case_no"]),
        "date": clean_text(record["judgment_date"]),
        "case_type": "妨害名譽（公然侮辱）",
        "court": clean_text(record["court"]),
        "judge": clean_text(record["judge"]),
        "sentence_type": clean_text(record["sentence_type"]),
        "sentence_value": clean_text(record["sentence_value"]),
        "judgment": clean_text(record["judgment"]),
        "original_crime_facts": clean_text(record["original_crime_facts"]),
        "original_sentencing": clean_text(record["original_sentencing"]),
    }

    return Document(
        page_content=page_content,
        metadata=metadata,
    )


def load_judgment_documents(
    jsonl_path: Path = JUDGMENTS_JSONL_PATH,
) -> list[Document]:
    """
    讀取可公開重建的判決 JSONL，轉換成 LangChain Document list。
    """

    documents: list[Document] = []

    with jsonl_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"SFT 同源判決資料第 {line_number} 行不是合法 JSON。"
                ) from exc

            documents.append(build_judgment_document(record))

    return documents

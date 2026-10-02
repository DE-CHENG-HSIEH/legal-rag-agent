"""把已核對的重要法律見解轉成可檢索且保留來源資訊的文件。"""

from pathlib import Path

import pandas as pd
from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTHORITIES_CSV_PATH = PROJECT_ROOT / "data/public/public_insult_authorities.csv"


def clean_text(value) -> str:
    """
    將 CSV 儲存格內容轉成乾淨字串。
    """

    if pd.isna(value):
        return ""

    return str(value).strip()


def build_authority_document(
    row: pd.Series,
) -> Document:
    """建立包含見解原文與可追溯來源 metadata 的檢索文件。"""

    topic = clean_text(row["主題"])

    authority_text = clean_text(row["判決意旨原文"])

    page_content = f"""
主題：
{topic}

判決意旨原文：
{authority_text}
""".strip()

    metadata = {
        "source_id": clean_text(row["資料編號"]),
        "citation": clean_text(row["案號或解釋字號"]),
        "date": clean_text(row["日期"]),
        "source_type": clean_text(row["來源類型"]),
        "original_location": clean_text(row["原文位置"]),
        "source_url": clean_text(row["來源連結"]),
        "verification_status": clean_text(row["核對狀態"]),
    }

    return Document(
        page_content=page_content,
        metadata=metadata,
    )


def load_authority_documents(
    csv_path: Path = AUTHORITIES_CSV_PATH,
) -> list[Document]:
    """
    讀取公然侮辱重要法律見解 CSV，
    轉換成 LangChain Document list。
    """

    dataframe = pd.read_csv(csv_path)

    documents = []

    for _, row in dataframe.iterrows():
        document = build_authority_document(row)

        documents.append(document)

    return documents

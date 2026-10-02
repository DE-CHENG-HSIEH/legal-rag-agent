"""從法務部全國法規資料庫取得受支援刑事法規的單一條文原文。"""

import re
import ssl
import unicodedata
from typing import TypedDict

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter


MOJ_LAW_SINGLE_URL = "https://law.moj.gov.tw/LawClass/LawSingle.aspx"


class LawConfig(TypedDict):
    """Static lookup metadata for one supported statute."""

    pcode: str
    aliases: set[str]


LAW_CONFIGS: dict[str, LawConfig] = {
    "中華民國刑法": {
        "pcode": "C0000001",
        "aliases": {
            "刑法",
            "中華民國刑法",
        },
    },
    "刑事訴訟法": {
        "pcode": "C0010001",
        "aliases": {
            "刑訴",
            "刑訴法",
            "刑事訴訟法",
        },
    },
    "刑事訴訟法施行法": {
        "pcode": "C0030006",
        "aliases": {
            "刑事訴訟法施行法",
            "刑訴施行法",
        },
    },
}


class MojSSLAdapter(HTTPAdapter):
    """
    專門提供給全國法規資料庫使用的 HTTPS Adapter。

    Python 3.13 預設啟用 VERIFY_X509_STRICT。
    全國法規資料庫目前的憑證鏈可能無法通過此嚴格檢查。

    這裡只停用 VERIFY_X509_STRICT，
    並沒有停用一般的 SSL 憑證驗證。
    """

    def init_poolmanager(
        self,
        connections,
        maxsize,
        block=False,
        **pool_kwargs,
    ):
        """Build a verified TLS context with only X509 strict mode relaxed."""

        # 保留 hostname 與憑證鏈驗證，只放寬 Python 3.13 新增、目前官方站
        # 憑證鏈無法通過的 STRICT 旗標；不得改成 verify=False。
        ssl_context = ssl.create_default_context()

        if hasattr(
            ssl,
            "VERIFY_X509_STRICT",
        ):
            ssl_context.verify_flags &= ~ssl.VERIFY_X509_STRICT

        pool_kwargs["ssl_context"] = ssl_context

        return super().init_poolmanager(
            connections,
            maxsize,
            block=block,
            **pool_kwargs,
        )


def create_moj_session() -> requests.Session:
    """
    建立專門用於法務部全國法規資料庫的
    requests Session。
    """

    session = requests.Session()

    session.mount(
        "https://law.moj.gov.tw/",
        MojSSLAdapter(),
    )

    return session


def normalize_law_name(
    law_name: str,
) -> str:
    """
    將使用者輸入的法規名稱或簡稱，
    統一轉換成正式法規名稱。
    """

    law_name = law_name.strip()

    for official_name, config in LAW_CONFIGS.items():
        aliases = config["aliases"]

        if law_name in aliases:
            return official_name

    supported_laws = "、".join(LAW_CONFIGS.keys())

    raise ValueError("目前此 Agent 僅支援以下核心刑事法規：" f"{supported_laws}")


def normalize_article_no(
    article_no: str,
) -> str:
    """
    將不同的法條條號表示方式，
    統一轉換成全國法規資料庫使用的格式。

    例如：

    309
    第309條
    309條

    都轉換為：

    309

    而：

    第15條之1
    15條之1
    15-1

    都轉換為：

    15-1
    """

    value = unicodedata.normalize(
        "NFKC",
        str(article_no),
    )

    value = re.sub(
        r"\s+",
        "",
        value,
    )

    value = value.replace(
        "－",
        "-",
    )

    value = value.replace(
        "—",
        "-",
    )

    pattern = re.compile(
        r"^"
        r"第?"
        r"(\d+)"
        r"(?:條)?"
        r"(?:"
        r"之(\d+)"
        r"|"
        r"-(\d+)"
        r")?"
        r"(?:第\d+項)?"
        r"$"
    )

    match = pattern.fullmatch(value)

    if match is None:
        raise ValueError(
            "無法辨識法條條號："
            f"{article_no}。"
            "請使用例如："
            "309、第309條、15-1、第15條之1。"
        )

    base_number = match.group(1)

    sub_number = match.group(2) or match.group(3)

    if sub_number:
        return f"{base_number}-{sub_number}"

    return base_number


def get_law_pcode(
    official_law_name: str,
) -> str:
    """
    根據正式法規名稱，
    取得全國法規資料庫使用的 PCode。
    """

    return LAW_CONFIGS[official_law_name]["pcode"]


def build_article_url(
    pcode: str,
    article_no: str,
) -> str:
    """
    建立全國法規資料庫的單一法條網址。
    """

    return f"{MOJ_LAW_SINGLE_URL}" f"?pcode={pcode}" f"&flno={article_no}"


def fetch_criminal_law_article(
    law_name: str,
    article_no: str,
) -> dict[str, str] | None:
    """
    即時連線至法務部全國法規資料庫，
    查詢指定刑事法規的單一法條原文。
    """

    official_law_name = normalize_law_name(law_name)

    normalized_article_no = normalize_article_no(article_no)

    pcode = get_law_pcode(official_law_name)

    url = build_article_url(
        pcode=pcode,
        article_no=normalized_article_no,
    )

    headers = {
        "User-Agent": ("Mozilla/5.0 " "Legal-RAG-Agent/1.0"),
    }

    session = create_moj_session()

    try:
        response = session.get(
            url,
            headers=headers,
            timeout=15,
        )

        response.raise_for_status()

    finally:
        session.close()

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    # Limit extraction to the requested article's DOM container. Scanning all
    # following page text also captures footer access keys and related links.
    article_content = ""
    for row in soup.select(".law-reg-content .row"):
        heading = row.select_one(".col-no")
        body = row.select_one(".col-data .law-article")
        if heading is None or body is None:
            continue
        heading_text = re.sub(r"\s+", "", heading.get_text())
        if heading_text != f"第{normalized_article_no}條":
            continue
        article_content = body.get_text("\n", strip=True)
        break

    if not article_content:
        return None

    return {
        "law_name": official_law_name,
        "article_no": normalized_article_no,
        "article_content": article_content,
        "source_url": url,
    }

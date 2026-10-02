"""連線官方法規網站，檢查支援法規、條號正規化與錯誤處理。"""

from app.tools.criminal_law_lookup import (
    lookup_criminal_law_article,
)
from scripts.manual_checks.runtime import build_manual_tool_runtime


TEST_CASES = [
    {
        "law_name": "刑法",
        "article_no": "第309條",
    },
    {
        "law_name": "刑訴法",
        "article_no": "第449條",
    },
]


def main() -> None:
    runtime = build_manual_tool_runtime()
    for index, test_case in enumerate(TEST_CASES, start=1):
        print("=" * 80)
        print(f"測試 {index}\n")
        result = lookup_criminal_law_article.invoke(
            {
                "law_name": test_case["law_name"],
                "article_no": test_case["article_no"],
                "runtime": runtime,
            }
        )
        print(f"{result}\n")


if __name__ == "__main__":
    main()

"""以代表性法律爭點檢查重要見解的召回與重排序品質。"""

from app.tools.authority_search import (
    search_public_insult_authorities,
)
from scripts.manual_checks.runtime import build_manual_tool_runtime


QUERIES = [
    "公然侮辱罪中的公然性應如何判斷？",
    "公然侮辱罪如何衡量言論自由與名譽權？",
    "哪些情況可能不構成公然侮辱罪？",
]


def main() -> None:
    runtime = build_manual_tool_runtime()
    for index, query in enumerate(QUERIES, start=1):
        print("=" * 80)
        print(f"測試 {index}")
        print(f"Query：{query}\n")
        result = search_public_insult_authorities.invoke(
            {"query": query, "runtime": runtime}
        )
        print(f"{result}\n")


if __name__ == "__main__":
    main()

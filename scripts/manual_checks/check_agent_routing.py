"""實際呼叫 Agent，檢查各類提示是否路由到預期的法律工具。"""

from collections import Counter
from uuid import uuid4

from langchain_core.messages import AIMessage

from app.agent.context import LegalAgentContext
from app.agent.legal_agent import get_legal_agent
from app.agent.schemas import normalize_markdown_text


TEST_CASES = [
    {
        "name": "一般對話，不應使用任何 Tool",
        "message": """
你好，請簡單說明你目前可以協助律師做哪些事情。
""".strip(),
        "expected_tools": set(),
    },
    {
        "name": "只查刑事法條",
        "message": """
請幫我查詢中華民國刑法第309條的現行法條原文。

這次只需要查法條，
不需要搜尋相似判決或重要法律見解。
""".strip(),
        "expected_tools": {
            "lookup_criminal_law_article",
        },
    },
    {
        "name": "只查重要法律見解",
        "message": """
請搜尋公然侮辱罪中，
法院或憲法裁判對「公然性」如何認定的重要法律見解。

這次不要搜尋一般相似案件，
也不需要查詢刑法條文。
""".strip(),
        "expected_tools": {
            "search_public_insult_authorities",
        },
    },
    {
        "name": "只搜尋相似判決",
        "message": """
請根據以下案件內容搜尋5筆相似的公然侮辱判決。

犯罪事實：
被告在711便利商店之公開場所與告訴人發生爭執，竟直接辱罵告訴人：「幹、幹你娘、幹你娘機掰、我去你媽的」。

量刑要素：
被告否認犯行、尚未向告訴人道歉、雙方亦未達成和解。
被告國中畢業、無業、小康之家庭經濟狀況、無需扶養之人。
被告無前科、素行良好。

這次只需要搜尋相似案件，
不需要另外查法條或重要法律見解。
""".strip(),
        "expected_tools": {
            "search_similar_judgments",
        },
    },
    {
        "name": "三個 Tool 都應使用",
        "message": """
我正在研究以下公然侮辱案件：

犯罪事實：
被告於便利商店內與告訴人發生爭執，
當著數名顧客的面，
以侮辱性言詞辱罵告訴人。

量刑要素：
被告否認犯行，
事後未向告訴人道歉，
雙方尚未達成和解，
被告無前科。

請完成以下三件事情：

1. 查詢中華民國刑法第309條的現行法條原文。
2. 搜尋公然侮辱罪相關的重要判決或憲法裁判法律見解。
3. 搜尋5筆與本案相似的一般公然侮辱判決。

最後再根據查詢結果進行整理，
並清楚區分：
- 官方法條原文
- 重要法律見解原文
- 相似判決原文
- AI分析
""".strip(),
        "expected_tools": {
            "lookup_criminal_law_article",
            "search_public_insult_authorities",
            "search_similar_judgments",
        },
    },
    {
        "name": "只執行 TAIDE LoRA 量刑預測",
        "message": """
請使用量刑預測模型評估以下公然侮辱案件。

犯罪事實：
被告在便利商店內當著數名顧客，以貶抑性言詞辱罵告訴人。

量刑因素：
被告坦承犯行，無前科，尚未與告訴人達成和解。

法院：臺灣臺北地方法院
預計裁判年份：民國114年
承審法官：王小明

這次只需要模型預測，不需要搜尋法條、重要見解或相似判決。
""".strip(),
        "expected_tools": {
            "predict_public_insult_sentence",
        },
    },
]


def extract_tool_names(
    messages,
) -> list[str]:
    """從 Agent message trace 依序取出模型發出的工具名稱。"""

    tool_names = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for tool_call in message.tool_calls:
            tool_name = tool_call.get("name")
            if tool_name:
                tool_names.append(tool_name)
    return tool_names


def run_test_case(
    agent,
    test_case: dict,
) -> bool:
    """以獨立 checkpoint thread 執行一筆真實 Agent 路由檢查。"""

    print("=" * 100)
    print(f"測試：{test_case['name']}")
    print()
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (test_case["message"]),
                }
            ]
        },
        config={"configurable": {"thread_id": str(uuid4())}},
        context=LegalAgentContext(),
    )
    messages = result["messages"]
    tool_names = extract_tool_names(messages)
    actual_tools = set(tool_names)
    expected_tools = test_case["expected_tools"]
    print("預期 Tool：", expected_tools)
    print("實際 Tool：", actual_tools)

    duplicate_tools: dict[str, int] = {}
    if tool_names:
        print("Tool 呼叫順序：", tool_names)
        tool_counts = Counter(tool_names)
        duplicate_tools = {
            name: count for name, count in tool_counts.items() if count > 1
        }
        if duplicate_tools:
            print("重複呼叫：", duplicate_tools)
    else:
        print("Tool 呼叫順序：無")

    passed = actual_tools == expected_tools and not duplicate_tools
    print()
    if passed:
        print("結果：PASS")
    else:
        print("結果：FAIL")
        missing_tools = expected_tools - actual_tools
        unexpected_tools = actual_tools - expected_tools
        if missing_tools:
            print("缺少 Tool：", missing_tools)
        if unexpected_tools:
            print("不應出現的 Tool：", unexpected_tools)

    print()
    print("Agent 最終回答：")
    print()
    structured = result.get("structured_response")
    if hasattr(structured, "answer_markdown"):
        answer = structured.answer_markdown
    elif isinstance(structured, dict):
        answer = structured.get("answer_markdown", "")
    else:
        answer = getattr(messages[-1], "content", "")
    print(normalize_markdown_text(answer))
    print()
    return passed


def main() -> None:
    """執行所有需要模型、工具與網路的路由案例。"""

    agent = get_legal_agent()
    passed_count = 0
    total_count = len(TEST_CASES)
    for test_case in TEST_CASES:
        passed = run_test_case(
            agent=agent,
            test_case=test_case,
        )
        if passed:
            passed_count += 1

    print("=" * 100)
    print("Tool Routing 測試完成")
    print(f"通過：{passed_count}/{total_count}")
    if passed_count == total_count:
        print("所有 Tool Routing 測試皆通過。")
    else:
        print("仍有 Tool Routing 規則需要調整。")


if __name__ == "__main__":
    main()

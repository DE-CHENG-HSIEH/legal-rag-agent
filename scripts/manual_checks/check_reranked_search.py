"""執行一筆相似判決 dense retrieval 與 cross-encoder reranking smoke test。"""

from app.rag.reranker import rerank_judgments
from app.rag.vector_store import get_judgment_vector_store


QUERY = """
犯罪事實：
被告在711便利商店之公開場所與告訴人發生爭執，竟直接辱罵告訴人：「幹、幹你娘、幹你娘機掰、我去你媽的」。

量刑要素：
被告否認犯行、尚未向告訴人道歉、雙方亦未達成和解。
被告國中畢業、無業、小康之家庭經濟狀況、無需扶養之人。
被告無前科、素行良好。
""".strip()


def main() -> None:
    vector_store = get_judgment_vector_store()
    initial_results = vector_store.similarity_search_with_score(query=QUERY, k=20)
    print(f"第一階段候選判決：{len(initial_results)} 筆")

    reranked_results = rerank_judgments(
        query=QUERY,
        docs_with_scores=initial_results,
        top_n=5,
    )
    print(f"第二階段重排結果：{len(reranked_results)} 筆")
    print("=" * 80)

    for index, (document, vector_score, rerank_score) in enumerate(
        reranked_results,
        start=1,
    ):
        print(f"重排後第 {index} 名")
        print(f"裁判字號：{document.metadata.get('case_no')}")
        print(f"處刑種類：{document.metadata.get('sentence_type')}")
        print(f"處刑輕重：{document.metadata.get('sentence_value')}")
        print(f"向量檢索分數：{vector_score:.4f}")
        print(f"重排序分數：{rerank_score:.4f}\n")
        print(document.page_content)
        print("=" * 80)


if __name__ == "__main__":
    main()

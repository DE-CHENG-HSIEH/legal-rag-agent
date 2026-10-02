"""相似判決的 dense retrieval 與 cross-encoder reranking pipeline。"""

from langchain_core.documents import Document

from app.rag.reranker import rerank_judgments
from app.rag.vector_store import get_judgment_vector_store


def retrieve_similar_judgments(
    query: str,
    initial_k: int = 20,
    top_n: int = 5,
) -> list[tuple[Document, float, float]]:
    """
    根據使用者輸入的犯罪事實與量刑要素，
    先使用向量搜尋取得候選判決，
    再使用 BGE reranker 重新排序。

    Args:
        query:
            使用者輸入的犯罪事實與量刑要素。

        initial_k:
            第一階段向量搜尋取得的候選判決數量。

        top_n:
            reranker 重排後保留的判決數量。

    Returns:
        list[tuple[Document, float, float]]

        每一筆資料依序包含：
        - Document
        - vector_score
        - rerank_score
    """

    query = query.strip()

    if not query:
        return []

    # 先擴大向量召回，再用 cross-encoder 收斂成可安全放入模型上下文的結果集。
    vector_store = get_judgment_vector_store()
    initial_results = vector_store.similarity_search_with_score(
        query=query,
        k=initial_k,
    )

    reranked_results = rerank_judgments(
        query=query,
        docs_with_scores=initial_results,
        top_n=top_n,
    )

    return reranked_results

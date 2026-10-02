"""重要法律見解的向量召回與重排序流程。"""

from langchain_core.documents import Document

from app.rag.reranker import (
    rerank_judgments,
)
from app.rag.vector_store import (
    get_authority_vector_store,
)


def _select_distinct_sources(
    results: list[tuple[Document, float, float]],
    *,
    top_n: int,
) -> list[tuple[Document, float, float]]:
    """Keep the highest-ranked passage from each authority citation."""

    selected = []
    seen_sources: set[str] = set()

    for result in results:
        document = result[0]
        citation = str(document.metadata.get("citation", "")).strip()
        source_id = str(document.metadata.get("source_id", "")).strip()
        source_key = citation or source_id

        if source_key in seen_sources:
            continue

        seen_sources.add(source_key)
        selected.append(result)
        if len(selected) == top_n:
            break

    return selected


def search_authorities(
    query: str,
    initial_k: int = 12,
    top_n: int = 5,
) -> list[tuple[Document, float, float]]:
    """
    搜尋公然侮辱的重要判決、
    憲法裁判與重要法律見解。

    第一階段：
        BGE-M3 + Chroma

    第二階段：
        bge-reranker-v2-m3
    """

    query = query.strip()

    if not query:
        return []

    # 與相似判決共用「先召回、再重排」契約，讓兩套法律語料的排序行為一致。
    vector_store = get_authority_vector_store()
    candidate_count = max(initial_k, top_n * 4)

    initial_results = vector_store.similarity_search_with_score(
        query=query,
        k=candidate_count,
    )

    # 一個裁判可能拆成數個可檢索段落。先將候選全部重排，再以案號保留最高分
    # 段落，避免回傳筆數被同一權威來源重複占用。
    reranked_results = rerank_judgments(
        query=query,
        docs_with_scores=initial_results,
        top_n=candidate_count,
    )

    return _select_distinct_sources(reranked_results, top_n=top_n)

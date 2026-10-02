"""以 cross-encoder 對向量召回候選重新排序。"""

import os
from threading import Lock
from typing import Optional

from FlagEmbedding import FlagReranker
from langchain_core.documents import Document


RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
DEFAULT_RAG_DEVICE = "cpu"
_reranker: Optional[FlagReranker] = None
_reranker_lock = Lock()


def get_reranker() -> FlagReranker:
    """建立 process-wide、執行緒安全的 BGE reranker 實例。"""

    global _reranker

    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                # CPU 上停用 fp16，避免未支援或精度不穩定；若切換至 GPU，
                # 是否採半精度應再依目標硬體量測後調整。
                _reranker = FlagReranker(
                    RERANKER_MODEL_NAME,
                    query_max_length=512,
                    passage_max_length=512,
                    use_fp16=False,
                    devices=os.getenv(
                        "RAG_DEVICE",
                        DEFAULT_RAG_DEVICE,
                    ),
                )

    return _reranker


def rerank_judgments(
    query: str,
    docs_with_scores: list[tuple[Document, float]],
    top_n: int = 5,
) -> list[tuple[Document, float, float]]:
    """
    使用 BGE reranker 重新排序候選判決。

    回傳格式：
    [
        (Document, vector_score, rerank_score),
        ...
    ]
    """

    if not docs_with_scores:
        return []

    pairs = []

    for document, vector_score in docs_with_scores:
        pairs.append([query, document.page_content])

    reranker = get_reranker()

    rerank_scores = reranker.compute_score(
        pairs,
        normalize=True,
    )

    combined_results = []

    for doc_with_score, rerank_score in zip(
        docs_with_scores,
        rerank_scores,
    ):
        document, vector_score = doc_with_score

        combined_results.append(
            (
                document,
                float(vector_score),
                float(rerank_score),
            )
        )

    reranked_results = sorted(
        combined_results,
        key=lambda item: item[2],
        reverse=True,
    )

    return reranked_results[:top_n]

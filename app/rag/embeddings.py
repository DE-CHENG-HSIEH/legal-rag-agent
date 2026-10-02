"""集中管理 BGE-M3 embedding 模型，避免 Streamlit rerun 重複載入。"""

import os
from threading import Lock
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_RAG_DEVICE = "cpu"
_embedding_model: Optional[HuggingFaceEmbeddings] = None
_embedding_model_lock = Lock()


def get_embedding_model() -> HuggingFaceEmbeddings:
    """以 double-checked lock 建立 process-wide、執行緒安全的模型實例。"""

    global _embedding_model

    if _embedding_model is None:
        with _embedding_model_lock:
            if _embedding_model is None:
                # RAG 預設留在 CPU，避免和 TAIDE 8B 同時競爭 Apple MPS 記憶體；
                # 有 CUDA 的部署環境仍可透過 RAG_DEVICE 明確覆寫。
                _embedding_model = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL_NAME,
                    model_kwargs={
                        "device": os.getenv(
                            "RAG_DEVICE",
                            DEFAULT_RAG_DEVICE,
                        ),
                    },
                    encode_kwargs={
                        "normalize_embeddings": True,
                    },
                )

    return _embedding_model

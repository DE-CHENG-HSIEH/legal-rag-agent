"""由版本控制語料自動建立、驗證並更新本機 Chroma 快取。"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.rag.authority_documents import (
    AUTHORITIES_CSV_PATH,
    load_authority_documents,
)
from app.rag.documents import JUDGMENTS_JSONL_PATH, load_judgment_documents
from app.rag.embeddings import get_embedding_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VECTOR_STORE_DIRECTORY = PROJECT_ROOT / "data/vector_store"

JUDGMENT_COLLECTION_NAME = "public_insult_sft_judgments_v1"
AUTHORITY_COLLECTION_NAME = "public_insult_authorities"
INDEX_SCHEMA_VERSION = "2"
INDEX_BATCH_SIZE = 128

_INDEX_LOCK = Lock()
_VECTOR_STORE_INIT_LOCK = Lock()
_judgment_vector_store: Chroma | None = None
_authority_vector_store: Chroma | None = None


def _source_digest(source_paths: tuple[Path, ...]) -> str:
    """Hash source content and index schema so stale local indexes are rebuilt."""

    digest = hashlib.sha256(INDEX_SCHEMA_VERSION.encode("utf-8"))
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"找不到公開檢索資料：{path}。請確認 GitHub 資料檔已完整下載。"
            )
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _read_digest(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def _write_digest(path: Path, value: str) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(value + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _synchronize_collection(
    *,
    vector_store: Any,
    source_paths: tuple[Path, ...],
    digest_path: Path,
    load_documents: Callable[[], list[Document]],
    build_document_id: Callable[[Document], str],
) -> bool:
    """Synchronize one Chroma collection with its version-controlled source.

    Chroma is a generated cache, not a release artifact. The source digest makes
    first-run setup automatic and also prevents updated public data from being
    served through stale metadata left in a developer's local database.
    """

    source_digest = _source_digest(source_paths)
    with _INDEX_LOCK:
        existing_ids = vector_store.get(include=[]).get("ids", [])
        if existing_ids and _read_digest(digest_path) == source_digest:
            return False

        documents = load_documents()
        document_ids = [build_document_id(document) for document in documents]
        if not documents:
            raise ValueError("公開檢索資料不可為空。")
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("公開檢索資料產生了重複的 Chroma document ID。")

        # Remove the manifest before mutation. A failed rebuild is retried on the
        # next request instead of being mistaken for a complete synchronized index.
        digest_path.unlink(missing_ok=True)
        if existing_ids:
            vector_store.delete(ids=existing_ids)

        for start in range(0, len(documents), INDEX_BATCH_SIZE):
            end = start + INDEX_BATCH_SIZE
            vector_store.add_documents(
                documents=documents[start:end],
                ids=document_ids[start:end],
            )

        _write_digest(digest_path, source_digest)
        return True


def _judgment_document_id(document: Document) -> str:
    case_id = str(document.metadata["case_id"])
    return hashlib.sha256(case_id.encode("utf-8")).hexdigest()


def _authority_document_id(document: Document) -> str:
    source_id = str(document.metadata["source_id"])
    citation = str(document.metadata["citation"])
    return hashlib.sha256(f"{source_id}|{citation}".encode("utf-8")).hexdigest()


def get_judgment_vector_store() -> Chroma:
    """Return one process-wide, source-synchronized judgment vector store."""

    global _judgment_vector_store

    if _judgment_vector_store is None:
        with _VECTOR_STORE_INIT_LOCK:
            if _judgment_vector_store is None:
                # Chroma keeps process-wide client state keyed by persistence path.
                # Serializing both collections prevents parallel first-use races.
                VECTOR_STORE_DIRECTORY.mkdir(parents=True, exist_ok=True)
                vector_store = Chroma(
                    collection_name=JUDGMENT_COLLECTION_NAME,
                    embedding_function=get_embedding_model(),
                    persist_directory=str(VECTOR_STORE_DIRECTORY),
                )
                _synchronize_collection(
                    vector_store=vector_store,
                    source_paths=(JUDGMENTS_JSONL_PATH,),
                    digest_path=VECTOR_STORE_DIRECTORY / "judgments.source.sha256",
                    load_documents=load_judgment_documents,
                    build_document_id=_judgment_document_id,
                )
                _judgment_vector_store = vector_store

    return _judgment_vector_store


def get_authority_vector_store() -> Chroma:
    """Return one process-wide, source-synchronized authority vector store."""

    global _authority_vector_store

    if _authority_vector_store is None:
        with _VECTOR_STORE_INIT_LOCK:
            if _authority_vector_store is None:
                VECTOR_STORE_DIRECTORY.mkdir(parents=True, exist_ok=True)
                vector_store = Chroma(
                    collection_name=AUTHORITY_COLLECTION_NAME,
                    embedding_function=get_embedding_model(),
                    persist_directory=str(VECTOR_STORE_DIRECTORY),
                )
                _synchronize_collection(
                    vector_store=vector_store,
                    source_paths=(AUTHORITIES_CSV_PATH,),
                    digest_path=VECTOR_STORE_DIRECTORY / "authorities.source.sha256",
                    load_documents=load_authority_documents,
                    build_document_id=_authority_document_id,
                )
                _authority_vector_store = vector_store

    return _authority_vector_store

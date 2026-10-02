"""驗證 Chroma 索引在來源異動與併發建置時保持一致。"""

import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from unittest.mock import patch

from langchain_core.documents import Document

from app.rag import vector_store
from app.rag.vector_store import _synchronize_collection


class FakeVectorStore:
    def __init__(self) -> None:
        self.ids: list[str] = []
        self.add_calls = 0
        self.delete_calls = 0

    def get(self, include: list[str]) -> dict[str, list[str]]:
        return {"ids": list(self.ids)}

    def add_documents(
        self,
        documents: list[Document],
        ids: list[str],
    ) -> None:
        self.add_calls += 1
        self.ids.extend(ids)

    def delete(self, ids: list[str]) -> None:
        self.delete_calls += 1
        self.ids = [document_id for document_id in self.ids if document_id not in ids]


class VectorStoreSynchronizationTests(unittest.TestCase):
    def setUp(self) -> None:
        vector_store._judgment_vector_store = None
        vector_store._authority_vector_store = None

    def tearDown(self) -> None:
        vector_store._judgment_vector_store = None
        vector_store._authority_vector_store = None

    def test_source_digest_skips_current_index_and_rebuilds_stale_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.jsonl"
            digest_path = root / "source.sha256"
            source_path.write_text('{"version": 1}\n', encoding="utf-8")
            fake_store = FakeVectorStore()
            load_count = 0

            def load_documents() -> list[Document]:
                nonlocal load_count
                load_count += 1
                return [
                    Document(page_content="公開裁判文字", metadata={"case_id": "1"})
                ]

            def build_document_id(document: Document) -> str:
                return str(document.metadata["case_id"])

            first_rebuild = _synchronize_collection(
                vector_store=fake_store,
                source_paths=(source_path,),
                digest_path=digest_path,
                load_documents=load_documents,
                build_document_id=build_document_id,
            )
            second_rebuild = _synchronize_collection(
                vector_store=fake_store,
                source_paths=(source_path,),
                digest_path=digest_path,
                load_documents=load_documents,
                build_document_id=build_document_id,
            )

            with patch.object(
                vector_store,
                "get_embedding_model_revision",
                return_value="new-revision",
            ):
                model_rebuild = _synchronize_collection(
                    vector_store=fake_store,
                    source_paths=(source_path,),
                    digest_path=digest_path,
                    load_documents=load_documents,
                    build_document_id=build_document_id,
                )

            source_path.write_text('{"version": 2}\n', encoding="utf-8")
            stale_rebuild = _synchronize_collection(
                vector_store=fake_store,
                source_paths=(source_path,),
                digest_path=digest_path,
                load_documents=load_documents,
                build_document_id=build_document_id,
            )

        self.assertTrue(first_rebuild)
        self.assertFalse(second_rebuild)
        self.assertTrue(model_rebuild)
        self.assertTrue(stale_rebuild)
        self.assertEqual(load_count, 3)
        self.assertEqual(fake_store.add_calls, 3)
        self.assertEqual(fake_store.delete_calls, 2)
        self.assertEqual(fake_store.ids, ["1"])

    def test_parallel_first_use_serializes_chroma_clients(self) -> None:
        creation_counts: dict[str, int] = {}
        active_clients = 0
        max_active_clients = 0
        count_lock = Lock()

        def create_chroma(*, collection_name: str, **kwargs):
            nonlocal active_clients, max_active_clients
            del kwargs
            with count_lock:
                active_clients += 1
                max_active_clients = max(max_active_clients, active_clients)
                creation_counts[collection_name] = (
                    creation_counts.get(collection_name, 0) + 1
                )
            time.sleep(0.05)
            with count_lock:
                active_clients -= 1
            return object()

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(
                    vector_store,
                    "VECTOR_STORE_DIRECTORY",
                    Path(directory),
                ),
                patch.object(vector_store, "Chroma", side_effect=create_chroma),
                patch.object(
                    vector_store, "get_embedding_model", return_value=object()
                ),
                patch.object(vector_store, "_synchronize_collection"),
            ):
                getters = [
                    vector_store.get_judgment_vector_store,
                    vector_store.get_authority_vector_store,
                ] * 4
                with ThreadPoolExecutor(max_workers=8) as executor:
                    stores = list(executor.map(lambda getter: getter(), getters))

        self.assertEqual(max_active_clients, 1)
        self.assertEqual(
            creation_counts,
            {
                vector_store.JUDGMENT_COLLECTION_NAME: 1,
                vector_store.AUTHORITY_COLLECTION_NAME: 1,
            },
        )
        self.assertIs(stores[0], stores[2])
        self.assertIs(stores[1], stores[3])


if __name__ == "__main__":
    unittest.main()

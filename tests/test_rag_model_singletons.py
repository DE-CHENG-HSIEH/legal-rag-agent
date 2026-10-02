"""驗證大型 embedding 與 reranker 模型在併發首次使用時只載入一次。"""

import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from unittest.mock import patch

from app.rag import embeddings, reranker


class RagModelSingletonTests(unittest.TestCase):
    def setUp(self) -> None:
        embeddings._embedding_model = None
        reranker._reranker = None

    def tearDown(self) -> None:
        embeddings._embedding_model = None
        reranker._reranker = None

    def test_embedding_model_is_created_once_under_concurrency(self) -> None:
        sentinel = object()
        creation_count = 0
        count_lock = Lock()

        def create_model(*args, **kwargs):
            nonlocal creation_count
            with count_lock:
                creation_count += 1
            time.sleep(0.05)
            return sentinel

        with patch.object(
            embeddings,
            "HuggingFaceEmbeddings",
            side_effect=create_model,
        ):
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(
                    executor.map(
                        lambda _: embeddings.get_embedding_model(),
                        range(8),
                    )
                )

        self.assertEqual(creation_count, 1)
        self.assertTrue(all(result is sentinel for result in results))

    def test_reranker_is_created_once_under_concurrency(self) -> None:
        sentinel = object()
        creation_count = 0
        count_lock = Lock()

        def create_model(*args, **kwargs):
            nonlocal creation_count
            with count_lock:
                creation_count += 1
            time.sleep(0.05)
            return sentinel

        with patch.object(
            reranker,
            "FlagReranker",
            side_effect=create_model,
        ):
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(
                    executor.map(
                        lambda _: reranker.get_reranker(),
                        range(8),
                    )
                )

        self.assertEqual(creation_count, 1)
        self.assertTrue(all(result is sentinel for result in results))


if __name__ == "__main__":
    unittest.main()

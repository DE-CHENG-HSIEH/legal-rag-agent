"""Agent 模型韌性 middleware 的組裝順序測試。"""

import unittest
from unittest.mock import Mock, patch

from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware

from app.agent import legal_agent


class AgentConfigurationTests(unittest.TestCase):
    def tearDown(self) -> None:
        legal_agent.get_legal_agent.cache_clear()

    def test_primary_model_retries_before_fallback(self) -> None:
        legal_agent.get_legal_agent.cache_clear()
        sentinel = object()
        create_agent = Mock(return_value=sentinel)

        with (
            patch.object(
                legal_agent, "ChatOpenAI", side_effect=lambda **kwargs: kwargs
            ),
            patch.object(legal_agent, "create_agent", create_agent),
        ):
            self.assertIs(legal_agent.get_legal_agent(), sentinel)

        middleware = create_agent.call_args.kwargs["middleware"]
        fallback_index = next(
            index
            for index, item in enumerate(middleware)
            if isinstance(item, ModelFallbackMiddleware)
        )
        retry_index = next(
            index
            for index, item in enumerate(middleware)
            if isinstance(item, ModelRetryMiddleware)
        )

        self.assertLess(fallback_index, retry_index)
        self.assertEqual(middleware[retry_index].on_failure, "error")


if __name__ == "__main__":
    unittest.main()

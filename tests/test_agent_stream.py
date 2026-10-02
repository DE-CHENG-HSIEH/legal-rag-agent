"""驗證背景 Agent 串流可傳遞事件並在使用者停止時取消。"""

import asyncio
import threading
import unittest

from ui.agent_stream import (
    consume_with_interrupts,
    run_in_current_event_loop,
    stream_in_background,
)


class StopSignal(BaseException):
    pass


class AgentStreamTests(unittest.TestCase):
    def test_stream_events_are_delivered(self) -> None:
        received = []
        cancelled = threading.Event()

        async def source():
            yield ("custom", "開始")
            yield ("values", {"messages": []})

        run_in_current_event_loop(
            consume_with_interrupts(
                source(),
                on_event=received.append,
                interrupt_point=lambda: None,
                on_cancel=cancelled.set,
                poll_interval=0.001,
            )
        )

        self.assertEqual(len(received), 2)
        self.assertFalse(cancelled.is_set())

    def test_interrupt_cancels_pending_stream(self) -> None:
        cancelled = threading.Event()
        source_finalized = threading.Event()

        async def source():
            try:
                await asyncio.sleep(10)
                yield "too late"
            finally:
                source_finalized.set()

        def raise_stop_signal() -> None:
            raise StopSignal()

        with self.assertRaises(StopSignal):
            run_in_current_event_loop(
                consume_with_interrupts(
                    source(),
                    on_event=lambda event: None,
                    interrupt_point=raise_stop_signal,
                    on_cancel=cancelled.set,
                    poll_interval=0.001,
                )
            )

        self.assertTrue(cancelled.is_set())
        self.assertTrue(source_finalized.is_set())

    def test_background_stream_keeps_interrupt_point_responsive(self) -> None:
        cancelled = threading.Event()
        interrupt_checks = 0

        async def source():
            await asyncio.sleep(0.03)
            yield ("custom", "完成")

        def count_interrupts() -> None:
            nonlocal interrupt_checks
            interrupt_checks += 1

        events = list(
            stream_in_background(
                source,
                cancellation_event=cancelled,
                interrupt_point=count_interrupts,
                poll_interval=0.005,
            )
        )

        self.assertEqual(events, [("custom", "完成")])
        self.assertGreater(interrupt_checks, 1)
        self.assertFalse(cancelled.is_set())


if __name__ == "__main__":
    unittest.main()

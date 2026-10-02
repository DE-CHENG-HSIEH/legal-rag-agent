"""讓 Streamlit 可以在等待 Agent 事件時持續處理停止請求。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
from contextlib import suppress
from queue import Empty, Queue
import threading
from typing import Any


class _CancellationRequested(BaseException):
    """結束背景串流，不把使用者取消誤報為 Agent 錯誤。"""


async def consume_with_interrupts(
    stream: AsyncIterator[Any],
    *,
    on_event: Callable[[Any], None],
    interrupt_point: Callable[[], None],
    on_cancel: Callable[[], None] | None = None,
    poll_interval: float = 0.2,
) -> None:
    """消費非同步串流，並定期讓 UI 檢查停止請求。"""

    iterator = stream.__aiter__()
    pending_event: asyncio.Future[Any] | None = None
    completed = False

    try:
        pending_event = asyncio.ensure_future(iterator.__anext__())

        while True:
            done, _ = await asyncio.wait(
                {pending_event},
                timeout=poll_interval,
            )

            interrupt_point()

            if not done:
                continue

            try:
                event = pending_event.result()
            except StopAsyncIteration:
                completed = True
                break

            on_event(event)
            pending_event = asyncio.ensure_future(iterator.__anext__())
    finally:
        if not completed and on_cancel is not None:
            on_cancel()

        if pending_event is not None and not pending_event.done():
            pending_event.cancel()
            with suppress(
                asyncio.CancelledError,
                StopAsyncIteration,
            ):
                await pending_event

        close_stream = getattr(iterator, "aclose", None)
        if close_stream is not None:
            with suppress(
                asyncio.CancelledError,
                RuntimeError,
            ):
                await close_stream()


def run_in_current_event_loop(
    coroutine: Coroutine[Any, Any, Any],
) -> Any:
    """在沒有既有 event loop 的同步執行緒中完成 coroutine。"""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("目前執行緒已有運作中的 asyncio event loop。")

    # asyncio.run 會負責關閉 loop；背景 worker 結束後不留下 selector 資源。
    return asyncio.run(coroutine)


def stream_in_background(
    stream_factory: Callable[[], AsyncIterator[Any]],
    *,
    cancellation_event: threading.Event,
    interrupt_point: Callable[[], None],
    poll_interval: float = 0.1,
) -> Iterator[Any]:
    """在背景執行 Agent，讓 Streamlit 主執行緒持續接收停止事件。

    Agent 或本機模型不能佔住 Streamlit script thread，否則前端雖然顯示
    停止按鈕，Python 端仍要等到模型回傳後才有機會處理中斷。
    """

    events: Queue[tuple[str, Any]] = Queue()

    def check_cancellation() -> None:
        if cancellation_event.is_set():
            raise _CancellationRequested()

    def run_worker() -> None:
        try:
            run_in_current_event_loop(
                consume_with_interrupts(
                    stream_factory(),
                    on_event=lambda event: events.put(("event", event)),
                    interrupt_point=check_cancellation,
                    on_cancel=cancellation_event.set,
                    poll_interval=poll_interval,
                )
            )
        except _CancellationRequested:
            pass
        except BaseException as error:
            events.put(("error", error))
        finally:
            events.put(("done", None))

    worker = threading.Thread(
        target=run_worker,
        name="legal-agent-stream",
        daemon=True,
    )
    worker.start()

    completed = False
    try:
        while not completed:
            # Streamlit 在 widget 呼叫點注入停止事件；保持輪詢才能即時收到。
            interrupt_point()
            try:
                kind, payload = events.get(timeout=poll_interval)
            except Empty:
                continue

            if kind == "event":
                yield payload
            elif kind == "error":
                raise payload
            elif kind == "done":
                completed = True
    finally:
        if not completed:
            cancellation_event.set()

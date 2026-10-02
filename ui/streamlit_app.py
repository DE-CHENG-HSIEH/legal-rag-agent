"""Streamlit 對話介面、任務範例與可中止的 Agent 執行生命週期。"""

import base64
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4


import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

APP_ICON_PATH = PROJECT_ROOT / "ui/assets/legal-ai-mark.png"
APP_HERO_PATH = PROJECT_ROOT / "ui/assets/legal-ai-hero.png"


from app.agent.context import (
    LegalAgentContext,
)

from app.agent.legal_agent import (
    get_legal_agent,
)
from app.agent.schemas import normalize_markdown_text
from app.agent.source_rendering import compose_final_answer
from ui.agent_stream import (
    stream_in_background,
)


st.set_page_config(
    page_title="公然侮辱案件研究與量刑輔助 Agent",
    page_icon=str(APP_ICON_PATH),
    layout="wide",
)


LOGGER = logging.getLogger(__name__)


# Keep presentation overrides centralized so the workflow remains native Streamlit
# and retains its built-in accessibility, rerun, and stop-button behavior.
APP_STYLES = """
<style>
    :root {
        --legal-ink: #292a2f;
        --legal-muted: #62646b;
        --legal-border: #dedfe2;
        --legal-surface: #f6f6f5;
        --legal-accent: #b4232c;
    }

    #MainMenu,
    footer,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {
        display: none !important;
    }

    header[data-testid="stHeader"] {
        min-height: 0;
        height: 0;
        background: transparent;
    }

    .block-container {
        max-width: 70rem;
        padding-top: 1.5rem;
        padding-bottom: 7.5rem;
    }

    .app-brand-title {
        margin: 0;
        color: var(--legal-ink);
        font-size: 2rem;
        font-weight: 720;
        line-height: 1.2;
        letter-spacing: 0;
    }

    .app-brand-summary {
        margin: 0.35rem 0 0;
        color: var(--legal-muted);
        font-size: 1rem;
        line-height: 1.65;
        letter-spacing: 0;
    }

    .app-hero {
        display: flex;
        min-height: 18.5rem;
        margin-bottom: 0.85rem;
        padding: 1.75rem 2rem;
        flex-direction: column;
        justify-content: center;
        background-position: center;
        background-repeat: no-repeat;
        background-size: cover;
    }

    .app-hero-title {
        max-width: 62%;
        margin: 0;
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 740;
        line-height: 1.2;
        letter-spacing: 0;
        white-space: nowrap;
        text-shadow: 0 2px 12px rgb(0 0 0 / 45%);
    }

    .app-hero-no-wrap {
        white-space: nowrap;
    }

    .app-hero-subtitle {
        max-width: 44%;
        margin: 0.7rem 0 0;
        color: #e9ecef;
        font-size: 1rem;
        line-height: 1.6;
        letter-spacing: 0;
        text-shadow: 0 1px 8px rgb(0 0 0 / 50%);
    }

    .app-disclaimer {
        margin: 0.75rem 0 1.4rem;
        padding-left: 0.75rem;
        border-left: 3px solid var(--legal-accent);
        color: var(--legal-muted);
        font-size: 0.9rem;
        line-height: 1.6;
        letter-spacing: 0;
    }

    div.stButton > button {
        min-height: 2.9rem;
        border: 1px solid var(--legal-border);
        border-radius: 6px;
        font-weight: 620;
        letter-spacing: 0;
    }

    div.stButton > button:hover {
        border-color: var(--legal-accent);
        color: var(--legal-accent);
    }

    [data-testid="stChatMessage"] {
        margin-bottom: 0.85rem;
        padding: 1rem 1.1rem;
        border: 1px solid #e6e6e8;
        border-radius: 8px;
        background: #ffffff;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
        line-height: 1.78;
        letter-spacing: 0;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2 {
        margin-top: 1.5rem;
        padding-bottom: 0.45rem;
        border-bottom: 1px solid var(--legal-border);
        color: var(--legal-ink);
        font-size: 1.35rem;
        letter-spacing: 0;
    }

    [data-testid="stChatInput"] {
        border-color: var(--legal-border);
        border-radius: 8px;
        box-shadow: 0 4px 18px rgb(31 33 38 / 7%);
    }

    @media (max-width: 640px) {
        .block-container {
            padding-top: 1rem;
            padding-right: 1rem;
            padding-left: 1rem;
        }

        .app-brand-title {
            font-size: 1.55rem;
        }

        .app-hero {
            min-height: 11.5rem;
            padding: 1.25rem;
            background-position: 58% center;
            background-size: cover;
        }

        .app-hero-title {
            max-width: 72%;
            font-size: 1.45rem !important;
            white-space: normal;
        }

        .app-hero-subtitle {
            max-width: 64%;
            font-size: 0.88rem;
        }
    }
</style>
"""


st.markdown(APP_STYLES, unsafe_allow_html=True)


STARTER_PROMPTS = {
    "搜尋相似判決": """請根據以下案件搜尋相似的公然侮辱判決，並比較犯罪事實、量刑因素與判決結果。

案件事實：
[請填寫事件經過、地點、公開情境與言詞或行為]

量刑因素：
[請填寫是否認罪、和解、賠償、前科、教育程度、工作與家庭狀況]

回傳筆數：
[請填寫 1 至 10；未填寫時為 5 筆]

請回傳相似判決，完整保留資料庫原文，並將判決原文與 AI 比較分析分開。""",
    "查詢法條與見解": """請針對以下法律問題查詢現行法條與重要法律見解。

法律問題或爭點：
[例如：刑法第 309 條、公然性的認定、侮辱性及言論自由權衡]

請將官方法條或法院見解原文與 AI 分析分開。""",
    "預測刑種與刑度群組": """請針對以下公然侮辱案件進行刑種與刑度群組預測。

犯罪事實：
[請填寫]

量刑因素：
[請填寫]

法院（選填）：
[已知時填寫完整法院名稱]

預計裁判年份（民國，選填）：
[已知時填寫]

法官姓名（選填）：
[已分案且已知時填寫]

若選填欄位未知，請直接使用 A 組訴前模式。請輸出刑種、刑度群組、參考量刑理由、使用版本與模型限制，不要將群組上下限表述為確切刑度。""",
    "進行完整案件分析": """請針對以下公然侮辱案件進行完整分析。

犯罪事實：
[請填寫]

量刑因素：
[請填寫]

法院、預計裁判年份與法官姓名（均選填）：
[已知多少填多少；未知時仍可使用 A 組訴前預測]

請依序查詢現行法條、重要法律見解與相似判決；犯罪事實與量刑因素具備時，即使用 TAIDE LoRA 預測刑種與刑度群組。法院、年份或法官未知不影響預測。最後整理有利因素、不利因素、案件差異與模型限制。""",
}

STARTER_ICONS = {
    "搜尋相似判決": ":material/search:",
    "查詢法條與見解": ":material/balance:",
    "預測刑種與刑度群組": ":material/analytics:",
    "進行完整案件分析": ":material/fact_check:",
}


CANCELLED_MESSAGE = (
    "已停止產生。你可以修改上一個提示後重新送出，" "或直接在下方輸入新的問題。"
)
ERROR_MESSAGE = "Agent 執行失敗，請確認模型、資料與網路設定後再試。"
OPENAI_KEY_PLACEHOLDERS = {
    "your_openai_api_key",
    "你的openai_api金鑰",
}


def require_openai_credentials() -> None:
    """Stop before Agent construction when the required API key is unavailable."""

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key and api_key not in OPENAI_KEY_PLACEHOLDERS:
        return

    st.error(
        "尚未設定 OpenAI API 金鑰。請先在專案根目錄的 `.env` 中填入 "
        "`OPENAI_API_KEY`，再重新啟動應用程式。"
    )
    st.stop()


@st.cache_resource(show_spinner=False)
def load_agent():
    """載入並跨 Streamlit rerun 重用同一個 Agent graph。"""

    return get_legal_agent()


def initialize_session_state() -> None:
    """初始化目前瀏覽器工作階段需要的狀態。"""

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid4())
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []
    if "rebuild_agent_thread" not in st.session_state:
        st.session_state.rebuild_agent_thread = False


def clear_conversation() -> None:
    """清除 UI 與 checkpoint 對話；新 thread_id 防止舊狀態回流。"""

    st.session_state.thread_id = str(uuid4())
    st.session_state.chat_history = []
    st.session_state.agent_messages = []
    st.session_state.pop("prompt_draft", None)
    st.session_state.rebuild_agent_thread = False


@st.cache_data(show_spinner=False)
def load_image_data_uri(path: str) -> str:
    """Encode a project image once so CSS can use it without an external host."""

    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_app_header() -> None:
    """Render an immersive empty state and a compact header during conversation."""

    if not st.session_state.chat_history:
        hero_uri = load_image_data_uri(str(APP_HERO_PATH))
        st.markdown(
            f"""
            <section class="app-hero" style="background-image: url('{hero_uri}')">
                <h1 class="app-hero-title">
                    公然侮辱案件
                    <span class="app-hero-no-wrap">
                        研究與量刑輔助 Agent
                    </span>
                </h1>
                <p class="app-hero-subtitle">
                    臺灣刑事法律研究、判決檢索與研究模型預測
                </p>
            </section>
            """,
            unsafe_allow_html=True,
        )
    else:
        logo_column, title_column, action_column = st.columns(
            [0.7, 7.9, 1.2],
            vertical_alignment="center",
        )
        with logo_column:
            st.image(str(APP_ICON_PATH), width=64)
        with title_column:
            st.markdown(
                """
                <div class="app-brand-title">
                    公然侮辱案件研究與量刑輔助 Agent
                </div>
                <div class="app-brand-summary">
                    臺灣刑事法律研究與研究模型預測
                </div>
                """,
                unsafe_allow_html=True,
            )
        with action_column:
            if st.button(
                "清除",
                icon=":material/delete_sweep:",
                help="清除目前對話",
                type="tertiary",
                use_container_width=True,
            ):
                clear_conversation()
                st.rerun()

    st.markdown(
        """
        <div class="app-disclaimer">
            系統會區分法律來源原文、研究模型預測與 AI 分析；
            結果僅供研究與專案展示，不構成法律意見或判決保證。
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_task_starters() -> str | None:
    """在空白對話顯示可編輯的任務提示範例。"""

    if st.session_state.chat_history:
        return None

    starter_container = st.empty()
    submitted_prompt = None

    with starter_container.container():
        st.subheader("開始任務")

        columns = st.columns(2)
        for index, (label, template) in enumerate(STARTER_PROMPTS.items()):
            with columns[index % 2]:
                if st.button(
                    label,
                    key=f"starter_{index}",
                    icon=STARTER_ICONS[label],
                    use_container_width=True,
                ):
                    st.session_state.prompt_draft = template

        if st.session_state.get("prompt_draft"):
            st.text_area(
                "提示草稿",
                key="prompt_draft",
                height=260,
            )

            if st.button(
                "送出提示",
                type="primary",
                use_container_width=True,
            ):
                prompt = st.session_state.prompt_draft.strip()
                if prompt:
                    submitted_prompt = prompt

    if submitted_prompt:
        starter_container.empty()

    return submitted_prompt


def render_chat_history() -> None:
    """只顯示整理後的 UI 紀錄，隔離 ToolMessage 與內部 tool calls。"""

    for item in st.session_state.chat_history:
        role = item.get("role")
        content = item.get("content", "")
        if not content:
            continue

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        elif role == "assistant":
            with st.chat_message("assistant", avatar=str(APP_ICON_PATH)):
                st.markdown(normalize_markdown_text(content))


def build_agent_config() -> dict:
    """建立 InMemorySaver 用來隔離瀏覽器對話的 thread config。"""

    return {"configurable": {"thread_id": (st.session_state.thread_id)}}


def build_agent_context(
    cancellation_event: threading.Event | None = None,
) -> LegalAgentContext:
    """建立不寫入對話 checkpoint 的單次執行狀態。"""

    return LegalAgentContext(cancellation_event=cancellation_event)


def get_final_answer_from_result(
    result: dict,
) -> str:
    """Combine model analysis with source artifacts rendered by application code."""

    return compose_final_answer(result)


def unpack_stream_event(
    event,
):
    """相容 ``(stream_mode, chunk)`` 與舊版裸 chunk 串流格式。"""

    if isinstance(event, tuple) and len(event) == 2:
        return event[0], event[1]
    return None, event


initialize_session_state()
render_app_header()

require_openai_credentials()
agent = load_agent()


starter_input = render_task_starters()


render_chat_history()


typed_input = st.chat_input(
    "輸入案件事實或法律研究需求……",
    submit_mode="stop",
)


user_input = starter_input or typed_input


if user_input:
    if (
        st.session_state.chat_history
        and st.session_state.chat_history[-1].get("role") == "user"
    ):
        st.session_state.thread_id = str(uuid4())
        st.session_state.rebuild_agent_thread = bool(st.session_state.agent_messages)
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": CANCELLED_MESSAGE,
            }
        )
        with st.chat_message("assistant", avatar=str(APP_ICON_PATH)):
            st.info(CANCELLED_MESSAGE)

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar=str(APP_ICON_PATH)):
        input_messages = []
        if st.session_state.rebuild_agent_thread:
            input_messages.extend(st.session_state.agent_messages)
        input_messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        agent_input = {"messages": input_messages}
        agent_config = build_agent_config()
        cancellation_event = threading.Event()
        agent_context = build_agent_context(cancellation_event)
        seen_status_messages: set[str] = set()

        try:
            progress_box = st.empty()
            with progress_box.container(border=True):
                st.markdown("**處理進度**")
                progress_messages = st.container()
                interrupt_probe = st.empty()

            # Mutable state lets the nested callback retain the latest LangGraph
            # values event without exposing partial ToolMessages in the UI.
            stream_state: dict[str, Any] = {"final_result": None}

            def handle_agent_event(event) -> None:
                """Render unique progress events and retain only the latest full state."""

                stream_mode, chunk = unpack_stream_event(event)
                if stream_mode == "custom":
                    status_message = str(chunk).strip()
                    if status_message and status_message not in seen_status_messages:
                        seen_status_messages.add(status_message)
                        with progress_messages:
                            st.write(status_message)

                elif stream_mode == "values":
                    stream_state["final_result"] = chunk
                elif isinstance(chunk, dict) and "messages" in chunk:
                    stream_state["final_result"] = chunk

            def check_for_stop_request() -> None:
                """Yield to Streamlit so its built-in stop button can interrupt this run."""

                interrupt_probe.empty()

            # 背景執行 Agent，前景持續觸碰 Streamlit API 以接收停止事件。
            for event in stream_in_background(
                lambda: agent.astream(
                    agent_input,
                    config=agent_config,
                    context=agent_context,
                    stream_mode=["custom", "values"],
                ),
                cancellation_event=cancellation_event,
                interrupt_point=check_for_stop_request,
            ):
                handle_agent_event(event)

            final_result = stream_state["final_result"]
            progress_box.empty()
            if final_result is None:
                st.error("Agent 沒有回傳最終結果。")
            else:
                st.session_state.agent_messages = final_result.get(
                    "messages",
                    [],
                )
                st.session_state.rebuild_agent_thread = False
                answer = get_final_answer_from_result(final_result)
                if not answer:
                    answer = "Agent 已完成處理，但沒有產生可顯示的回答。"
                st.markdown(answer)
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

        except Exception:
            LOGGER.exception("Legal Agent execution failed")
            st.error(ERROR_MESSAGE)
            # Persist an ordinary failure as an assistant turn. Otherwise the next
            # user message would be mistaken for recovery from a cancelled run.
            st.session_state.chat_history.append(
                {"role": "assistant", "content": ERROR_MESSAGE}
            )
        finally:
            # Streamlit 停止本輪時也會進入 finally，通知背景 Agent 與 TAIDE。
            cancellation_event.set()

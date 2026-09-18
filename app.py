"""PaperReader-RAG Streamlit 前端入口。

这个文件负责把后端能力展示成可操作页面：
- 左侧：上传论文、建立索引、选择当前论文和多论文对比范围；
- Agent 工作台：输入自然语言指令，由 ResearchAgent 自动路由到工具；
- 论文问答：保留原始 RAG 问答链路，方便学习和对比；
- 阅读笔记：单独生成当前论文阅读卡片；
- 多论文对比与导出：生成对比表，并导出当前 Agent 记忆。
"""

from __future__ import annotations

from pathlib import Path
import re

import streamlit as st

from backend.config import PAPER_DIR, settings
from backend.embeddings import OpenAICompatibleClient
from backend.exporter import Exporter
from backend.ieee_search import (
    LiteratureSearchService,
    format_search_answer,
)
from backend.memory import AgentMemory
from backend.router import AgentRouter
from backend.services import PaperService
from backend.tools import LibraryStatusTool, SourceExplainTool
from backend.ui import apply_app_style, render_top_navigation
from backend.vector_store import ChromaVectorStore


st.set_page_config(
    page_title="PaperReader-RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_app_style()


@st.cache_resource
def get_vector_store() -> ChromaVectorStore:
    """创建并缓存 Chroma 向量库连接。

    Streamlit 每次用户点击按钮都会从头执行脚本。使用 cache_resource 可以避免
    每次交互都重新创建数据库客户端。
    """

    return ChromaVectorStore()


@st.cache_resource
def get_llm_client() -> OpenAICompatibleClient:
    """创建并缓存 OpenAI-compatible 客户端。

    客户端本身是可复用资源；真正的 API 调用只会在 embed_texts/chat 时发生。
    """

    return OpenAICompatibleClient()


@st.cache_resource
def get_paper_service() -> PaperService:
    """Return the UI-independent service also used by the FastAPI app."""

    service = PaperService(get_vector_store(), get_llm_client())
    if settings.enable_startup_warmup:
        service.warmup()
    return service


@st.cache_resource
def get_literature_search_service() -> LiteratureSearchService:
    """Create the IEEE search boundary once per Streamlit process."""

    llm_client = get_llm_client() if settings.is_ready else None
    return LiteratureSearchService(llm_client=llm_client)


def save_uploaded_pdf(uploaded_file) -> Path:
    """把 Streamlit 上传的 PDF 保存到 data/papers。

    保存到本地后，PDF Loader 才能用统一的 Path 方式读取文件。
    """

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    file_path = PAPER_DIR / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())
    return file_path


@st.cache_data(show_spinner=False, max_entries=64)
def get_pdf_page_count(path_text: str, modified_ns: int) -> int:
    """Return a cached page count while invalidating when the PDF changes."""

    del modified_ns
    import fitz

    with fitz.open(path_text) as document:
        return document.page_count


@st.cache_data(show_spinner=False, max_entries=48)
def render_pdf_page(path_text: str, modified_ns: int, page_index: int) -> bytes:
    """Render one PDF page instead of loading the complete file in the browser."""

    del modified_ns
    import fitz

    with fitz.open(path_text) as document:
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.35, 1.35), alpha=False)
        return pixmap.tobytes("png")


def get_agent_memory() -> AgentMemory:
    """从 session_state 中获取当前浏览器会话的 AgentMemory。

    Streamlit 是脚本式运行模型，普通 Python 变量会在每次交互时重新初始化。
    session_state 可以保存跨交互状态，所以 Memory 必须放在这里。
    """

    if "agent_memory" not in st.session_state:
        st.session_state.agent_memory = AgentMemory()
    return st.session_state.agent_memory


def render_sources(sources: list[dict], *, key_prefix: str = "sources") -> None:
    """把检索来源渲染到页面上。

    sources 来自向量库检索结果，包含文本、文件名、页码和 distance。
    前端这里只展示短文本，完整结构仍保留在后端结果中。
    """

    if not sources:
        return
    with st.expander("查看检索到的引用片段", expanded=False):
        for index, source in enumerate(sources, start=1):
            metadata = source.get("metadata", {})
            section = metadata.get("section_path") or metadata.get("section_title") or "未识别章节"
            page = str(metadata.get("parent_pages") or metadata.get("page") or "?")
            file_name = metadata.get("file_name") or "unknown"
            section_type = metadata.get("section_type") or "unknown"
            text = str(source.get("generation_text") or source.get("text") or "").replace("\n", " ")
            physical_pages = [int(value) for value in re.findall(r"\d+", page)]
            source_pdf = PAPER_DIR / file_name
            st.markdown(f"**[S{index}] {section}**")
            st.caption(f"{file_name} · PDF 第 {page} 页 · {section_type}")
            st.write(text[:600])
            if physical_pages and source_pdf.exists():
                if st.button(
                    f"在右侧查看 PDF 第 {physical_pages[0]} 页",
                    key=f"jump_{key_prefix}_{index}_{file_name}_{physical_pages[0]}",
                    width="stretch",
                ):
                    st.session_state.pending_pdf_navigation = {
                        "file_name": file_name,
                        "page": physical_pages[0],
                    }
                    st.rerun()


def render_answer_details(result: dict, *, paper_id: str | None) -> None:
    """Show compact, presentation-friendly runtime and trust information."""

    refusal_reason = result.get("refusal_reason")
    status_labels = {
        None: "有依据回答",
        "no_sources": "未检索到来源",
        "model_refusal": "模型主动拒答",
        "missing_citation": "缺少引用，系统拒答",
        "invalid_citation": "引用越界，系统拒答",
    }
    timings = result.get("timings_ms") or {}
    model_metadata = result.get("model_metadata") or {}
    model_used = model_metadata.get("model_used") or "未知"
    scope = "当前论文" if paper_id else "全部论文"
    top_k = result.get("top_k") or "?"
    st.caption(
        f"状态：{status_labels.get(refusal_reason, refusal_reason or '有依据回答')} · "
        f"范围：{scope} · Top-{top_k} · 模型：{model_used} · "
        f"检索：{float(timings.get('retrieval') or 0) / 1000:.2f}s · "
        f"生成：{float(timings.get('generation') or 0) / 1000:.2f}s · "
        f"总计：{float(timings.get('total') or 0) / 1000:.2f}s"
    )
    if result.get("citation_repaired"):
        st.info("模型生成了越界引用；系统已移除无效编号并保留有效引用。")
    if result.get("warning"):
        st.warning(result["warning"])


def compact_result_for_history(result: dict) -> dict:
    """Keep chat history responsive by dropping large retrieval-only payloads."""

    keep = {
        "answer",
        "refusal_reason",
        "refusal_type",
        "warning",
        "citation_repaired",
        "citation_retry_attempted",
        "model_metadata",
        "timings_ms",
        "top_k",
    }
    compact = {key: result.get(key) for key in keep if key in result}
    compact["sources"] = [
        {
            "text": str(source.get("generation_text") or source.get("text") or "")[:800],
            "metadata": {
                key: source.get("metadata", {}).get(key)
                for key in (
                    "file_name",
                    "page",
                    "parent_pages",
                    "section_path",
                    "section_title",
                    "section_type",
                )
            },
        }
        for source in result.get("sources", [])
    ]
    return compact


def ensure_api_ready() -> bool:
    """检查模型 API 配置是否完整。

    建立索引需要 embedding，问答/总结/对比需要 chat model，因此这些操作前都要
    确认 .env 已配置。导出已有结果不需要模型调用，所以不走这个检查。
    """

    if settings.is_ready:
        return True
    st.warning(
        "请分别配置生成模型 LLM_API_KEY/LLM_BASE_URL/LLM_MODEL，"
        "以及向量模型 EMBEDDING_API_KEY/EMBEDDING_BASE_URL/EMBEDDING_MODEL。"
    )
    return False


def ensure_ieee_ready() -> bool:
    if settings.ieee_is_ready:
        return True
    st.warning("请先在 .env 中配置 IEEE_API_KEY，然后重启应用。")
    return False


def render_literature_results(payload: dict, *, key_prefix: str) -> None:
    """Render verified IEEE metadata and the transparent ranking breakdown."""

    results = payload.get("results") or []
    plan = payload.get("query_plan") or {}
    years = [str(value) for value in (plan.get("start_year"), plan.get("end_year")) if value]
    year_label = "–".join(years) if years else "不限年份"
    st.caption(
        f"IEEE Xplore · 检索式：{plan.get('querytext') or '-'} · {year_label} · "
        f"命中 {payload.get('total_records', 0)} 条，展示 {len(results)} 条"
    )
    if not results:
        st.info("没有找到满足当前主题和年份条件、且带 DOI 或稳定 IEEE 链接的结果。")
        return

    for item in results:
        with st.container(border=True, key=f"{key_prefix}_paper_{item.get('rank')}_{item.get('article_number') or item.get('doi') or item.get('api_position')}"):
            st.markdown(f"**{item.get('rank')}. {item.get('title')}**")
            authors = item.get("authors") or []
            author_text = ", ".join(authors[:6]) or "作者信息未返回"
            if len(authors) > 6:
                author_text += " 等"
            st.caption(
                f"{author_text} · {item.get('venue') or 'IEEE'} · {item.get('year') or '年份未知'}"
            )
            score = item.get("score") or {}
            st.caption(
                f"综合 {score.get('total', 0)}/12 · 相关性 {score.get('relevance', 0)}/5 · "
                f"来源匹配 {score.get('venue_fit', 0)}/3 · 时效性 {score.get('recency', 0)}/2 · "
                f"元数据 {score.get('metadata', 0)}/2"
            )
            if item.get("abstract_snippet"):
                st.write(item["abstract_snippet"])
            link_columns = st.columns([1, 1, 3])
            if item.get("ieee_url"):
                link_columns[0].link_button("IEEE Xplore", item["ieee_url"], width="stretch")
            if item.get("doi"):
                link_columns[1].link_button(
                    "DOI", f"https://doi.org/{item['doi']}", width="stretch"
                )

    with st.expander("查看检索记录", expanded=False):
        search_log = payload.get("search_log") or {}
        st.json(
            {
                "自然语言需求": payload.get("request"),
                "实际检索式": search_log.get("querytext"),
                "来源": search_log.get("source_order"),
                "检索时间": search_log.get("searched_at"),
                "去重规则": search_log.get("deduplication"),
            }
        )


def run_agent_query(user_query: str) -> None:
    """执行 Agent 工作台的一条指令，并把结果写入 Memory。

    这里单独做一层前端调度，是为了区分“需要模型 API 的任务”和“不需要 API 的任务”。
    问答、总结、对比需要 embedding/chat；来源解释和导出只读 Memory，可以离线执行。
    """

    intent = AgentRouter().route(user_query)

    if intent == "export":
        file_format = Exporter.pick_format(user_query)
        agent_memory.add_turn("user", user_query)
        path = Exporter().export_memory(agent_memory, file_format=file_format)
        answer = f"已导出到：{path}"
        agent_memory.last_export_path = str(path)
        agent_memory.remember_result(answer, [])
        agent_memory.add_turn("assistant", answer)
        return

    if intent == "source_explain":
        agent_memory.add_turn("user", user_query)
        result = SourceExplainTool().run(agent_memory.last_sources)
        agent_memory.remember_result(result.answer, result.sources)
        agent_memory.add_turn("assistant", result.answer)
        return

    if intent == "library_status":
        agent_memory.add_turn("user", user_query)
        result = LibraryStatusTool(get_vector_store()).run(user_query)
        agent_memory.remember_result(result.answer, [])
        agent_memory.add_turn("assistant", result.answer)
        return

    if intent == "literature_search" and not ensure_ieee_ready():
        return

    if not ensure_api_ready():
        return

    # Agent 会自动完成：意图识别 -> 工具调用 -> 结果检查 -> 写回 Memory。
    with st.spinner("Agent 正在判断意图并调用工具..."):
        result = get_paper_service().run_agent(user_query, agent_memory)
    if result["warnings"]:
        for warning in result["warnings"]:
            st.warning(warning)


def index_paths(jobs: list[tuple[str, Path]]) -> bool:
    """Index uploaded or already-local PDFs with one shared progress display."""

    if not jobs or not ensure_api_ready():
        return False
    total_chunks = 0
    progress_bar = st.progress(0.0)
    progress_text = st.empty()
    stage_labels = {
        "parsing": "解析 PDF",
        "splitting": "识别章节并切分",
        "embedding": "生成向量",
        "writing": "写入数据库",
        "done": "完成",
    }
    stage_ranges = {
        "parsing": (0.00, 0.15),
        "splitting": (0.15, 0.10),
        "embedding": (0.25, 0.65),
        "writing": (0.90, 0.09),
        "done": (1.00, 0.00),
    }
    for file_index, (file_name, file_path) in enumerate(jobs):
        def update_progress(stage: str, current: int, total: int, *, _index=file_index, _name=file_name) -> None:
            start, width = stage_ranges.get(stage, (0.0, 0.0))
            ratio = min(max(current / max(total, 1), 0.0), 1.0)
            overall = (_index + min(start + width * ratio, 1.0)) / len(jobs)
            progress_bar.progress(overall)
            detail = f" {current}/{total}" if total > 1 else ""
            progress_text.caption(
                f"{_index + 1}/{len(jobs)} · {_name} · {stage_labels.get(stage, stage)}{detail}"
            )

        try:
            result = get_paper_service().index_pdf(file_path, progress_callback=update_progress)
        except Exception as exc:
            progress_text.error(f"{file_name} 索引失败：{exc}")
            return False
        total_chunks += result.chunk_count
    progress_bar.progress(1.0)
    st.session_state.index_notice = f"索引完成，共写入 {total_chunks} 个 chunks。"
    return True


def render_chat_turn(message: dict, index: int) -> None:
    """Render user questions on the right and assistant answers on the left."""

    role = message["role"]
    if role == "user":
        _, bubble = st.columns([1.15, 4])
        with bubble:
            st.markdown('<div class="rag-message-label user">你</div>', unsafe_allow_html=True)
            with st.container(border=True, key=f"chat_user_{index}"):
                st.markdown(message["content"])
        return

    bubble, _ = st.columns([4, 1.15])
    with bubble:
        st.markdown('<div class="rag-message-label assistant">PaperReader</div>', unsafe_allow_html=True)
        with st.container(border=True, key=f"chat_assistant_{index}"):
            st.markdown(message["content"])
            if message.get("result"):
                if message["result"].get("kind") == "literature_search":
                    render_literature_results(
                        message["result"]["payload"], key_prefix=f"history_search_{index}"
                    )
                else:
                    render_answer_details(message["result"], paper_id=message.get("paper_id"))
                    render_sources(message["result"].get("sources", []), key_prefix=f"history_{index}")


def render_chat_panel(paper_id: str | None, has_papers: bool) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if st.session_state.pop("clear_workspace_question", False):
        st.session_state.workspace_question = ""

    title_col, action_col = st.columns([4, 1])
    title_col.subheader("论文问答")
    if action_col.button("清空", disabled=not st.session_state.messages, width="stretch"):
        st.session_state.messages = []
        st.rerun()

    with st.container(height=610, border=True, key="chat_history"):
        if not st.session_state.messages:
            st.info("可对已上传论文提问，也可直接说“帮我查找近三年关于 RAG 的 IEEE 论文”。")
        for index, message in enumerate(st.session_state.messages):
            render_chat_turn(message, index)

        pending_question = st.session_state.get("pending_question")
        if pending_question:
            pending_intent = AgentRouter().route(pending_question)
            live_answer, _ = st.columns([4, 1.15])
            with live_answer:
                st.markdown(
                    '<div class="rag-message-label assistant">PaperReader</div>',
                    unsafe_allow_html=True,
                )
                with st.container(border=True, key="chat_assistant_live"):
                    answer_placeholder = st.empty()
                    if pending_intent == "literature_search":
                        try:
                            with st.spinner("正在理解检索需求并查询 IEEE Xplore..."):
                                search_payload = get_literature_search_service().search(pending_question)
                            answer = format_search_answer(search_payload)
                        except Exception as exc:
                            st.session_state.pop("pending_question", None)
                            st.error(f"文献检索失败：{exc}")
                        else:
                            st.session_state.pop("pending_question", None)
                            answer_placeholder.markdown(answer)
                            render_literature_results(search_payload, key_prefix="live_search")
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": answer,
                                    "result": {"kind": "literature_search", "payload": search_payload},
                                }
                            )
                            st.rerun()
                    elif pending_intent == "library_status":
                        try:
                            status_result = LibraryStatusTool(get_vector_store()).run(pending_question)
                        except Exception as exc:
                            st.session_state.pop("pending_question", None)
                            st.error(f"知识库状态查询失败：{exc}")
                        else:
                            st.session_state.pop("pending_question", None)
                            answer_placeholder.markdown(status_result.answer)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": status_result.answer}
                            )
                            st.rerun()
                    elif pending_intent == "corpus_analysis":
                        try:
                            with st.spinner("正在读取全库背景章节、执行语义聚类并生成类别名称..."):
                                corpus_result = get_paper_service().run_agent(
                                    pending_question, agent_memory
                                )
                        except Exception as exc:
                            st.session_state.pop("pending_question", None)
                            st.error(f"全库分类失败：{exc}")
                        else:
                            st.session_state.pop("pending_question", None)
                            answer_placeholder.markdown(corpus_result["answer"])
                            st.session_state.messages.append(
                                {"role": "assistant", "content": corpus_result["answer"]}
                            )
                            st.rerun()
                    else:
                        streamed_answer = ""
                        result = None
                        try:
                            for event in get_paper_service().answer_stream(pending_question, paper_id=paper_id):
                                if event["event"] == "delta":
                                    streamed_answer += event["data"].get("text", "")
                                    answer_placeholder.markdown(streamed_answer + "▌")
                                elif event["event"] == "done":
                                    result = event["data"]
                                elif event["event"] == "error":
                                    raise RuntimeError(event["data"].get("error", "流式回答失败"))
                        except Exception as exc:
                            st.session_state.pop("pending_question", None)
                            st.error(f"问答失败：{exc}")
                        else:
                            st.session_state.pop("pending_question", None)
                            if result is None:
                                st.error("问答失败：生成服务没有返回最终结果。")
                            else:
                                answer_placeholder.markdown(result["answer"])
                                render_answer_details(result, paper_id=paper_id)
                                render_sources(result.get("sources", []), key_prefix="live")
                                st.session_state.messages.append(
                                    {
                                        "role": "assistant",
                                        "content": result["answer"],
                                        "result": compact_result_for_history(result),
                                        "paper_id": paper_id,
                                    }
                                )
                                st.rerun()

    quick1, quick2, quick3 = st.columns(3)
    if quick1.button("核心方法", width="stretch", key="chat_method"):
        st.session_state.workspace_question = "这篇论文的核心方法和主要创新是什么？"
    if quick2.button("实验结论", width="stretch", key="chat_result"):
        st.session_state.workspace_question = "这篇论文的主要实验结论是什么？"
    if quick3.button("局限性", width="stretch", key="chat_limit"):
        st.session_state.workspace_question = "论文有哪些局限性和可改进方向？"

    question = st.text_area(
        "输入问题",
        key="workspace_question",
        placeholder="针对当前论文提问，或直接描述想查找的 IEEE 文献…",
        height=76,
        label_visibility="collapsed",
    )
    if st.button(
        "发送",
        type="primary",
        disabled=not question.strip(),
        width="stretch",
        key="workspace_send",
    ):
        clean_question = question.strip()
        intent = AgentRouter().route(clean_question)
        if intent == "literature_search" and not ensure_ieee_ready():
            return
        if intent not in {"literature_search", "library_status"} and not has_papers:
            st.warning("当前没有已索引论文；如需外部检索，请明确说“查找/搜索 IEEE 论文”。")
            return
        if intent in {"literature_search", "library_status"} or ensure_api_ready():
            st.session_state.messages.append({"role": "user", "content": clean_question})
            st.session_state.pending_question = clean_question
            st.session_state.clear_workspace_question = True
            st.rerun()


vector_store = get_vector_store()
agent_memory = get_agent_memory()
papers = vector_store.list_papers()
paper_options = {paper["file_name"]: paper["paper_id"] for paper in papers}
local_pdfs = sorted(PAPER_DIR.glob("*.pdf"), key=lambda path: path.name.lower()) if PAPER_DIR.exists() else []

pending_pdf_navigation = st.session_state.pop("pending_pdf_navigation", None)
if pending_pdf_navigation:
    target_file = pending_pdf_navigation.get("file_name")
    target_page = int(pending_pdf_navigation.get("page") or 1)
    if target_file in paper_options:
        st.session_state.workspace_selected_file = target_file
        st.session_state.workspace_scope = "当前论文"
        st.session_state[f"pdf_page_{target_file}"] = target_page

if "main_navigation" not in st.session_state:
    st.session_state.main_navigation = "Chat"

navigation = render_top_navigation(st.session_state.main_navigation)

with st.sidebar:
    st.markdown('<div class="rag-sidebar-title">文件工作区</div>', unsafe_allow_html=True)
    st.caption("上传论文并选择当前阅读范围")
    uploads = st.file_uploader(
        "快速上传 PDF",
        type=["pdf"],
        accept_multiple_files=True,
        key="workspace_upload",
    )
    if st.button("上传并建立索引", type="primary", disabled=not uploads, width="stretch"):
        jobs = []
        for upload in uploads or []:
            path = save_uploaded_pdf(upload)
            jobs.append((upload.name, path))
        if index_paths(jobs):
            st.rerun()

    st.divider()
    if paper_options:
        selected_file = st.selectbox(
            "当前 PDF",
            list(paper_options),
            key="workspace_selected_file",
        )
        scope = st.radio(
            "检索范围",
            ["当前论文", "全部论文"],
            key="workspace_scope",
        )
        selected_paper_id = paper_options[selected_file] if scope == "当前论文" else None
        selected_pdf_path = PAPER_DIR / selected_file
        selected_chunks = next(
            (paper["chunk_count"] for paper in papers if paper["file_name"] == selected_file),
            0,
        )
        st.markdown(
            f'<div class="rag-file-status"><b>{selected_chunks}</b> chunks<br>'
            f'<span>{"单篇 Top-5" if selected_paper_id else "多篇 Top-10"}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        selected_file = None
        selected_paper_id = None
        selected_pdf_path = None
        st.warning("当前没有已索引论文。")

    with st.expander(f"文件列表 · {len(papers)} 篇", expanded=True):
        if papers:
            for paper in papers:
                active = paper["file_name"] == selected_file
                marker = "●" if active else "○"
                st.markdown(
                    f'<div class="rag-file-row {"active" if active else ""}">'
                    f'<span>{marker}</span><div>{paper["file_name"]}'
                    f'<small>{paper["chunk_count"]} chunks</small></div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("上传后文件会显示在这里。")

    st.caption(f"本地 PDF：{len(local_pdfs)} · 已索引：{len(papers)}")
    st.divider()
    st.page_link("pages/2_多模态论文阅读.py", label="多模态论文阅读", width="stretch")

if st.session_state.pop("index_notice", None):
    st.success("论文索引已更新。")

if navigation == "Chat":
    st.markdown(
        '<div class="rag-view-heading"><h2>论文对话</h2>'
        '<p>基于论文证据提问，并在右侧同步核对 PDF 原文</p></div>',
        unsafe_allow_html=True,
    )
    agent_memory.set_scope(selected_paper_id, [])
    center, right = st.columns([1, 1.18], gap="large")
    with center:
        render_chat_panel(selected_paper_id, bool(papers))
    with right:
        with st.container(height=830, border=True):
            st.subheader("PDF 原文")
            if selected_pdf_path and selected_pdf_path.exists():
                st.caption(selected_file)
                try:
                    pdf_stat = selected_pdf_path.stat()
                    page_count = get_pdf_page_count(str(selected_pdf_path), pdf_stat.st_mtime_ns)
                    page_number = st.number_input(
                        "页码",
                        min_value=1,
                        max_value=max(page_count, 1),
                        step=1,
                        key=f"pdf_page_{selected_file}",
                    )
                    page_image = render_pdf_page(
                        str(selected_pdf_path),
                        pdf_stat.st_mtime_ns,
                        int(page_number) - 1,
                    )
                    st.image(page_image, width="stretch")
                    st.caption(f"第 {page_number} / {page_count} 页 · 可用页码控件快速跳转")
                except Exception as exc:
                    st.error(f"PDF 预览失败：{exc}")
            elif selected_file:
                st.warning("索引存在，但本地 PDF 文件未找到。")
            else:
                st.info("选择或上传论文后，可在这里对照查看原文。")

elif navigation == "Search":
    st.markdown(
        '<div class="rag-view-heading"><h2>智能文献检索</h2>'
        '<p>用自然语言描述研究主题、时间范围或方法，系统将查询 IEEE Xplore 并给出可核验结果</p></div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        with st.form("literature_search_form"):
            search_request = st.text_area(
                "检索需求",
                placeholder="例如：查找 2022 年以来关于多模态 RAG 用于科研论文理解的 IEEE 文献",
                height=110,
            )
            result_limit = st.slider("返回数量", min_value=5, max_value=25, value=10, step=5)
            submitted = st.form_submit_button("搜索 IEEE Xplore", type="primary", width="stretch")
        if submitted:
            if not search_request.strip():
                st.warning("请输入文献检索需求。")
            elif ensure_ieee_ready():
                try:
                    with st.spinner("正在生成检索式并查询 IEEE Xplore..."):
                        st.session_state.literature_search_result = get_literature_search_service().search(
                            search_request, limit=result_limit
                        )
                except Exception as exc:
                    st.error(f"文献检索失败：{exc}")

    if st.session_state.get("literature_search_result"):
        payload = st.session_state.literature_search_result
        st.markdown(format_search_answer(payload))
        render_literature_results(payload, key_prefix="search_page")
    elif not settings.ieee_is_ready:
        st.info("Search 页面已接入完成。请在 .env 中填写 IEEE_API_KEY 后重启应用。")

elif navigation == "Files":
    st.subheader("文件管理")
    upload_col, library_col = st.columns([1, 1.6], gap="large")
    with upload_col:
        with st.container(border=True):
            st.markdown("#### 上传新文件")
            uploads = st.file_uploader(
                "PDF 文件",
                type=["pdf"],
                accept_multiple_files=True,
                key="files_upload",
            )
            if st.button("上传并索引", type="primary", disabled=not uploads, width="stretch", key="files_index"):
                jobs = []
                for upload in uploads or []:
                    path = save_uploaded_pdf(upload)
                    jobs.append((upload.name, path))
                if index_paths(jobs):
                    st.rerun()

        indexed_names = set(paper_options)
        unindexed = [path for path in local_pdfs if path.name not in indexed_names]
        with st.container(border=True):
            st.markdown("#### 索引已有 PDF")
            local_selection = st.multiselect(
                "本地尚未进入当前向量库的文件",
                [path.name for path in unindexed],
                key="local_unindexed_files",
            )
            if st.button(
                "为所选文件建立索引",
                disabled=not local_selection,
                width="stretch",
                key="index_local_files",
            ):
                selected_paths = [(name, PAPER_DIR / name) for name in local_selection]
                if index_paths(selected_paths):
                    st.rerun()

    with library_col:
        metrics = st.columns(3)
        metrics[0].metric("已索引论文", len(papers))
        metrics[1].metric("向量 Chunks", vector_store.count_chunks())
        metrics[2].metric("本地 PDF", len(local_pdfs))
        st.caption(f"Collection：{settings.chroma_collection} · Embedding：{settings.embedding_model}")
        if papers:
            st.dataframe(
                [
                    {
                        "文件名": paper["file_name"],
                        "Paper ID": paper["paper_id"],
                        "Chunks": paper["chunk_count"],
                        "PDF 状态": "存在" if (PAPER_DIR / paper["file_name"]).exists() else "缺失",
                    }
                    for paper in papers
                ],
                hide_index=True,
                width="stretch",
                height=620,
            )
        else:
            st.info("当前 collection 尚未建立论文索引。")

else:
    st.subheader("阅读工具")
    if not papers:
        st.info("请先在 Files 栏目上传并索引论文。")
    selected_tool_file = st.selectbox(
        "当前论文",
        list(paper_options),
        key="tool_selected_file",
        disabled=not papers,
    ) if papers else None
    selected_tool_id = paper_options.get(selected_tool_file) if selected_tool_file else None
    compare_labels = st.multiselect(
        "多论文对比范围",
        list(paper_options),
        default=list(paper_options)[:2],
        key="tool_compare_files",
        disabled=not papers,
    )
    selected_compare_ids = [paper_options[label] for label in compare_labels]
    agent_memory.set_scope(selected_tool_id, selected_compare_ids)

    tab_agent, tab_note, tab_compare, tab_debug = st.tabs(
        ["Agent", "阅读笔记", "多论文对比与导出", "检索调试"]
    )
    with tab_agent:
        for turn in agent_memory.history:
            with st.chat_message(turn.role):
                st.markdown(turn.content)
        agent_query = st.text_input(
            "Agent 指令",
            placeholder="总结论文 / 解释刚才的来源 / 导出 Markdown",
            key="agent_query",
        )
        if st.button("执行 Agent 指令", type="primary", disabled=not agent_query or not papers):
            run_agent_query(agent_query)
            st.rerun()

    with tab_note:
        if st.button("生成当前论文阅读笔记", disabled=not selected_tool_id, type="primary"):
            if ensure_api_ready():
                with st.spinner("正在生成论文阅读笔记..."):
                    result = get_paper_service().summarize(selected_tool_id)
                st.markdown(result["note"])
                render_sources(result["sources"], key_prefix="summary")

    with tab_compare:
        if st.button("生成多论文对比表", disabled=len(selected_compare_ids) < 2, type="primary"):
            if ensure_api_ready():
                with st.spinner("正在生成缺失的阅读卡片并进行对比..."):
                    result = get_paper_service().run_agent("比较选中的多篇论文，输出对比表", agent_memory)
                st.markdown(result["answer"])
        export_col1, export_col2 = st.columns(2)
        if export_col1.button("导出 Markdown", width="stretch"):
            path = Exporter().export_memory(agent_memory, file_format="markdown")
            agent_memory.last_export_path = str(path)
            st.success(f"已导出到：{path}")
        if export_col2.button("导出 JSON", width="stretch"):
            path = Exporter().export_memory(agent_memory, file_format="json")
            agent_memory.last_export_path = str(path)
            st.success(f"已导出到：{path}")

    with tab_debug:
        st.write("查看 Chroma chunks、章节 metadata 和混合检索排序结果。")
        st.page_link("pages/1_向量数据库学习.py", label="打开向量数据库学习页", width="stretch")

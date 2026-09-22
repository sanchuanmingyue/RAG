"""PaperReader-RAG Streamlit 前端入口。

这个文件负责把后端能力展示成可操作页面：
- 左侧：上传论文、建立索引、选择当前论文和多论文对比范围；
- Chat：统一完成意图识别、论文问答、文献搜索和论文工具调度；
- 阅读工具：提供阅读笔记、多论文对比、导出和检索调试快捷入口；
- 多模态开关：在 Chat 中按需切换到 RAG-Anything 问答。
"""

from __future__ import annotations

import html
import json
from pathlib import Path
import re

import streamlit as st

from backend.config import PAPER_DIR, settings
from backend.arxiv_mcp import (
    ArxivSearchService,
    choose_literature_source,
    format_arxiv_search_answer,
)
from backend.conversation import ConversationResolver
from backend.conversation_store import ConversationStore
from backend.embeddings import OpenAICompatibleClient
from backend.exporter import Exporter
from backend.ieee_search import (
    LiteratureSearchService,
    format_search_answer,
)
from backend.memory import AgentMemory
from backend.multimodal_service import MultimodalPaperService, multimodal_index_ready
from backend.services import PaperService
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


@st.cache_resource
def get_arxiv_search_service() -> ArxivSearchService:
    llm_client = get_llm_client() if settings.llm_is_ready else None
    return ArxivSearchService(llm_client=llm_client)


@st.cache_resource
def get_conversation_store() -> ConversationStore:
    return ConversationStore()


@st.cache_resource
def get_multimodal_service() -> MultimodalPaperService:
    """Initialize RAG-Anything only after the user enables and uses it."""

    return MultimodalPaperService()


def activate_conversation(conversation_id: str) -> None:
    """Load one persisted conversation into the current Streamlit session."""

    store = get_conversation_store()
    st.session_state.conversation_id = conversation_id
    st.session_state.agent_memory = store.load_memory(conversation_id)
    st.session_state.messages = store.load_messages(conversation_id)
    st.session_state.pop("pending_question", None)
    st.session_state.pop("workspace_question", None)


def ensure_active_conversation() -> str:
    store = get_conversation_store()
    conversation_id = st.session_state.get("conversation_id")
    if not conversation_id or not store.conversation_exists(conversation_id):
        conversations = store.list_conversations(limit=1)
        conversation_id = (
            conversations[0]["id"] if conversations else store.create_conversation()
        )
        activate_conversation(conversation_id)
    return conversation_id


def persist_chat_exchange(
    *,
    user_content: str,
    assistant_content: str,
    intent: str,
    memory: AgentMemory,
    rewritten_query: str | None = None,
    assistant_result: dict | None = None,
    paper_id: str | None = None,
) -> None:
    get_conversation_store().save_exchange(
        ensure_active_conversation(),
        user_content=user_content,
        assistant_content=assistant_content,
        memory=memory,
        intent=intent,
        rewritten_query=rewritten_query,
        assistant_result=assistant_result,
        paper_id=paper_id,
    )


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

    ensure_active_conversation()
    if "agent_memory" not in st.session_state:
        activate_conversation(st.session_state.conversation_id)
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
    answer_mode = "深度综合" if result.get("long_form") else "快速问答"
    st.caption(
        f"状态：{status_labels.get(refusal_reason, refusal_reason or '有依据回答')} · "
        f"模式：{answer_mode} · 范围：{scope} · Top-{top_k} · 模型：{model_used} · "
        f"检索：{float(timings.get('retrieval') or 0) / 1000:.2f}s · "
        f"生成：{float(timings.get('generation') or 0) / 1000:.2f}s · "
        f"总计：{float(timings.get('total') or 0) / 1000:.2f}s"
    )
    if result.get("citation_repaired"):
        st.info("模型生成了越界引用；系统已移除无效编号并保留有效引用。")
    if result.get("warning"):
        st.warning(result["warning"])
    if result.get("conversation_rewritten_query"):
        with st.expander("查看追问解析", expanded=False):
            st.write(result["conversation_rewritten_query"])


def compact_result_for_history(result: dict) -> dict:
    """Keep chat history responsive by dropping large retrieval-only payloads."""

    keep = {
        "answer",
        "refusal_reason",
        "refusal_type",
        "warning",
        "citation_repaired",
        "citation_retry_attempted",
        "answer_quality_retry_attempted",
        "answer_quality_retry_reason",
        "model_metadata",
        "timings_ms",
        "top_k",
        "query_analysis",
        "conversation_rewritten_query",
        "long_form",
        "evidence_plan_used",
        "continuation_attempted",
        "max_output_tokens",
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


def ensure_arxiv_ready() -> bool:
    if settings.arxiv_mcp_is_ready:
        return True
    st.warning("arXiv MCP 未启用，请检查 ARXIV_MCP_ENABLED、COMMAND 和 ARGS。")
    return False


def ensure_literature_source_ready(source: str) -> bool:
    return ensure_arxiv_ready() if source == "arxiv" else ensure_ieee_ready()


def ensure_multimodal_ready() -> bool:
    if not settings.is_ready or not settings.vision_is_ready:
        st.warning("多模态问答需要完整配置文本、向量和视觉模型。")
        return False
    if not multimodal_index_ready():
        st.warning("多模态模式尚无可查询索引，请先到多模态索引管理页解析文档。")
        return False
    return True


def format_multimodal_result(result: object) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, (dict, list)):
        return f"```json\n{json.dumps(result, ensure_ascii=False, indent=2)}\n```"
    return str(result) if result is not None else "多模态模型未返回内容。"


def render_literature_results(payload: dict, *, key_prefix: str) -> None:
    """Render normalized IEEE or arXiv MCP literature metadata."""

    if str(payload.get("source") or "").lower().startswith("arxiv"):
        render_arxiv_results(payload, key_prefix=key_prefix)
        return

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


def render_arxiv_results(payload: dict, *, key_prefix: str) -> None:
    results = payload.get("results") or []
    plan = payload.get("query_plan") or {}
    st.caption(
        f"arXiv MCP · 工具：search_papers · 检索式：{plan.get('querytext') or '-'} · "
        f"命中 {payload.get('total_records', 0)} 条，展示 {len(results)} 条"
    )
    if not results:
        st.info("arXiv 没有返回满足当前条件的论文。")
        return
    for item in results:
        identity = item.get("arxiv_id") or item.get("rank")
        with st.container(border=True, key=f"{key_prefix}_arxiv_{identity}"):
            st.markdown(f"**{item.get('rank')}. {item.get('title')}**")
            authors = item.get("authors") or []
            st.caption(
                f"{', '.join(authors[:6]) or '作者信息未返回'} · "
                f"{item.get('year') or '年份未知'} · "
                f"{', '.join((item.get('categories') or [])[:5]) or '未分类'}"
            )
            if item.get("abstract_snippet"):
                st.write(item["abstract_snippet"])
            links = st.columns([1, 1, 3])
            if item.get("arxiv_url"):
                links[0].link_button("arXiv", item["arxiv_url"], width="stretch")
            if item.get("pdf_url"):
                links[1].link_button("PDF", item["pdf_url"], width="stretch")
    with st.expander("查看 MCP 调用记录", expanded=False):
        st.json(
            {
                "自然语言需求": payload.get("request"),
                "实际检索式": (payload.get("search_log") or {}).get("querytext"),
                "MCP Server": (payload.get("mcp") or {}).get("server"),
                "Tool": (payload.get("mcp") or {}).get("tool"),
                "Transport": (payload.get("search_log") or {}).get("transport"),
                "参数": (payload.get("mcp") or {}).get("arguments"),
            }
        )


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
                result_kind = message["result"].get("kind")
                if result_kind == "literature_search":
                    render_literature_results(
                        message["result"]["payload"], key_prefix=f"history_search_{index}"
                    )
                elif result_kind == "agent":
                    st.caption(f"已自动识别意图：{message['result'].get('intent') or '-'}")
                    workflow = message["result"].get("workflow") or {}
                    if workflow:
                        with st.expander("查看执行轨迹", expanded=False):
                            st.json(workflow.get("trace") or [])
                elif result_kind == "multimodal":
                    st.caption(
                        f"多模态 RAG-Anything · {message['result'].get('mode', 'hybrid')} 模式"
                    )
                else:
                    render_answer_details(message["result"], paper_id=message.get("paper_id"))


def _latest_chat_result() -> dict:
    for message in reversed(st.session_state.get("messages") or []):
        result = message.get("result")
        if message.get("role") == "assistant" and isinstance(result, dict):
            return result
    return {}


def render_information_panel(
    *,
    selected_file: str | None,
    selected_pdf_path: Path | None,
    memory: AgentMemory,
) -> None:
    """Render the latest evidence and PDF in a Kotaemon-style side panel."""

    sources = memory.last_sources or []
    latest_result = _latest_chat_result()
    with st.container(height=830, border=True, key="information_panel"):
        st.markdown('<div class="rag-info-title">信息面板</div>', unsafe_allow_html=True)
        scope_label = "当前论文" if memory.current_paper_id else "全部论文"
        if st.session_state.get("multimodal_chat_enabled"):
            mode_label = "多模态"
        elif memory.current_paper_id is None or st.session_state.get("single_paper_deep_enabled"):
            mode_label = "深度思考"
        else:
            mode_label = "快速问答"
        st.markdown(
            f'<div class="rag-info-subtitle">{scope_label} · {mode_label} · '
            f'{len(sources)} 条可核验证据</div>',
            unsafe_allow_html=True,
        )

        evidence_tab, pdf_tab = st.tabs(["证据", "PDF 原文"])
        with evidence_tab:
            if latest_result and latest_result.get("kind") not in {
                "literature_search",
                "agent",
                "multimodal",
            }:
                timings = latest_result.get("timings_ms") or {}
                status = "可核验" if not latest_result.get("refusal_reason") else "已保守拒答"
                st.caption(
                    f"{status} · Top-{latest_result.get('top_k') or '?'} · "
                    f"检索 {float(timings.get('retrieval') or 0) / 1000:.2f}s · "
                    f"总计 {float(timings.get('total') or 0) / 1000:.2f}s"
                )
            if not sources:
                st.info("完成一次论文问答后，这里会集中展示引用章节、页码和检索评分。")
            for index, source in enumerate(sources, start=1):
                metadata = source.get("metadata") or {}
                section = metadata.get("section_path") or metadata.get("section_title") or "未识别章节"
                page = str(metadata.get("parent_pages") or metadata.get("page") or "?")
                file_name = str(metadata.get("file_name") or "unknown")
                text = str(source.get("generation_text") or source.get("text") or "").replace("\n", " ")
                retrieval_source = str(source.get("retrieval_source") or "hybrid")
                reranker_score = source.get("section_reranker_score_normalized")
                section_score = source.get("section_score") or source.get("rerank_score")
                distance = source.get("distance")
                score_pills = [f"{retrieval_source}"]
                if isinstance(reranker_score, (int, float)):
                    score_pills.append(f"rerank {reranker_score:.3f}")
                elif isinstance(section_score, (int, float)):
                    score_pills.append(f"score {section_score:.3f}")
                if isinstance(distance, (int, float)):
                    score_pills.append(f"distance {distance:.3f}")
                pills = "".join(
                    f'<span class="rag-score-pill">{html.escape(value)}</span>'
                    for value in score_pills
                )
                with st.container(border=True, key=f"evidence_card_{index}"):
                    st.markdown(
                        f'<div class="rag-evidence-heading">[S{index}] {html.escape(str(section))}</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"{file_name} · PDF 第 {page} 页")
                    st.markdown(f'<div class="rag-score-row">{pills}</div>', unsafe_allow_html=True)
                    st.write(text[:520] or "该来源没有可显示的文本片段。")
                    physical_pages = [int(value) for value in re.findall(r"\d+", page)]
                    if physical_pages and (PAPER_DIR / file_name).exists():
                        if st.button(
                            f"定位到第 {physical_pages[0]} 页",
                            key=f"info_jump_{index}_{file_name}_{physical_pages[0]}",
                            width="stretch",
                        ):
                            st.session_state.pending_pdf_navigation = {
                                "file_name": file_name,
                                "page": physical_pages[0],
                            }
                            st.rerun()

        with pdf_tab:
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
                    st.caption(f"第 {page_number} / {page_count} 页")
                except Exception as exc:
                    st.error(f"PDF 预览失败：{exc}")
            elif selected_file:
                st.warning("索引存在，但本地 PDF 文件未找到。")
            else:
                st.info("选择或上传论文后，可在这里对照查看原文。")


def render_chat_panel(paper_id: str | None, has_papers: bool) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if st.session_state.pop("clear_workspace_question", False):
        st.session_state.workspace_question = ""

    multi_paper_qa = bool(has_papers and paper_id is None)
    title_col, deep_col, mode_col, action_col = st.columns([2.1, 1.45, 1.25, 0.8])
    title_col.subheader("论文助手")
    with deep_col:
        if multi_paper_qa:
            deep_enabled = st.toggle(
                "深度思考",
                value=True,
                disabled=True,
                key="multi_paper_deep_indicator",
                help="多篇论文问答会自动执行完整检索与证据规划。",
            )
        else:
            deep_enabled = st.toggle(
                "深度思考",
                key="single_paper_deep_enabled",
                help="单篇论文开启后，将执行多角度检索、证据规划和长回答。",
            )
    with mode_col:
        multimodal_enabled = st.toggle(
            "多模态",
            key="multimodal_chat_enabled",
            help="启用后，普通论文问题将改用 RAG-Anything 的图片、表格和公式索引。",
        )
    if action_col.button("清空", disabled=not st.session_state.messages, width="stretch"):
        get_conversation_store().clear_conversation(ensure_active_conversation())
        st.session_state.messages = []
        st.session_state.agent_memory = AgentMemory()
        st.rerun()

    if multimodal_enabled:
        if multimodal_index_ready():
            st.caption("多模态已启用：普通论文问题将使用 RAG-Anything hybrid 检索。")
        else:
            st.warning("多模态已启用，但尚未建立可查询索引。")
            st.page_link(
                "pages/2_多模态论文阅读.py",
                label="打开多模态索引管理",
                width="stretch",
            )
    elif multi_paper_qa:
        st.caption("当前为多篇论文范围：已自动开启深度思考和完整证据规划。")
    elif deep_enabled:
        st.caption("深度思考已启用：将扩大检索范围并先规划证据，再生成详细回答。")

    with st.container(height=610, border=True, key="chat_history"):
        if not st.session_state.messages:
            st.info("可以提问、总结、对比或导出，也可直接说“帮我查找近两年关于 RAG 的论文”。")
        for index, message in enumerate(st.session_state.messages):
            render_chat_turn(message, index)

        pending_question = st.session_state.get("pending_question")
        if pending_question:
            resolver = ConversationResolver(get_llm_client() if settings.is_ready else None)
            resolution = resolver.resolve(pending_question, agent_memory.state)
            pending_intent = resolution.intent
            live_answer, _ = st.columns([4, 1.15])
            with live_answer:
                st.markdown(
                    '<div class="rag-message-label assistant">PaperReader</div>',
                    unsafe_allow_html=True,
                )
                with st.container(border=True, key="chat_assistant_live"):
                    answer_placeholder = st.empty()
                    if pending_intent == "literature_search":
                        provider = choose_literature_source(pending_question)
                        try:
                            source_label = "arXiv MCP" if provider == "arxiv" else "IEEE Xplore"
                            with st.spinner(f"正在理解检索需求并查询 {source_label}..."):
                                search_request = resolution.rewritten_query or pending_question
                                if provider == "arxiv":
                                    search_payload = get_arxiv_search_service().search(search_request)
                                    answer = format_arxiv_search_answer(search_payload)
                                else:
                                    search_payload = get_literature_search_service().search(search_request)
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
                            agent_memory.remember_result(
                                answer,
                                [],
                                intent="literature_search",
                                user_query=pending_question,
                                rewritten_query=resolution.rewritten_query if resolution.is_followup else None,
                                artifacts={"search": search_payload},
                            )
                            persist_chat_exchange(
                                user_content=pending_question,
                                assistant_content=answer,
                                intent="literature_search",
                                memory=agent_memory,
                                rewritten_query=(
                                    resolution.rewritten_query if resolution.is_followup else None
                                ),
                                assistant_result={"kind": "literature_search", "payload": search_payload},
                            )
                            st.rerun()
                    elif pending_intent == "qa" and multimodal_enabled:
                        try:
                            with st.spinner("正在使用多模态索引检索文本、图片、表格和公式..."):
                                multimodal_raw = get_multimodal_service().ask(
                                    resolution.rewritten_query or pending_question,
                                    mode="hybrid",
                                )
                                multimodal_answer = format_multimodal_result(multimodal_raw)
                        except Exception as exc:
                            st.session_state.pop("pending_question", None)
                            st.error(f"多模态问答失败：{exc}")
                        else:
                            st.session_state.pop("pending_question", None)
                            answer_placeholder.markdown(multimodal_answer)
                            history_result = {"kind": "multimodal", "mode": "hybrid"}
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": multimodal_answer,
                                    "result": history_result,
                                    "paper_id": paper_id,
                                }
                            )
                            agent_memory.remember_result(
                                multimodal_answer,
                                [],
                                intent="qa",
                                user_query=pending_question,
                                rewritten_query=(
                                    resolution.rewritten_query if resolution.is_followup else None
                                ),
                            )
                            persist_chat_exchange(
                                user_content=pending_question,
                                assistant_content=multimodal_answer,
                                intent="qa",
                                memory=agent_memory,
                                rewritten_query=(
                                    resolution.rewritten_query if resolution.is_followup else None
                                ),
                                assistant_result=history_result,
                                paper_id=paper_id,
                            )
                            st.rerun()
                    elif pending_intent != "qa":
                        try:
                            with st.spinner("正在识别意图并调用对应论文工具..."):
                                agent_result = get_paper_service().run_agent(
                                    pending_question, agent_memory
                                )
                        except Exception as exc:
                            st.session_state.pop("pending_question", None)
                            st.error(f"任务执行失败：{exc}")
                        else:
                            st.session_state.pop("pending_question", None)
                            answer_placeholder.markdown(agent_result["answer"])
                            workflow = (agent_result.get("artifacts") or {}).get("workflow") or {}
                            history_result = {
                                "kind": "agent",
                                "intent": agent_result.get("intent"),
                                "warnings": agent_result.get("warnings") or [],
                                "workflow": workflow,
                            }
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": agent_result["answer"],
                                    "result": history_result,
                                    "paper_id": paper_id,
                                }
                            )
                            persist_chat_exchange(
                                user_content=pending_question,
                                assistant_content=agent_result["answer"],
                                intent=agent_result["intent"],
                                memory=agent_memory,
                                rewritten_query=(
                                    resolution.rewritten_query if resolution.is_followup else None
                                ),
                                assistant_result=history_result,
                                paper_id=paper_id,
                            )
                            for warning in agent_result.get("warnings") or []:
                                st.warning(warning)
                            st.rerun()
                    else:
                        streamed_answer = ""
                        result = None
                        progress_status = st.status("准备问答…", expanded=False)
                        try:
                            for event in get_paper_service().answer_stream(
                                pending_question,
                                paper_id=paper_id,
                                deep_mode=deep_enabled,
                                retrieval_question=(
                                    resolution.rewritten_query if resolution.is_followup else None
                                ),
                            ):
                                if event["event"] == "stage":
                                    stage_data = event.get("data") or {}
                                    progress_status.update(
                                        label=str(stage_data.get("message") or stage_data.get("label") or "处理中…"),
                                        state="running",
                                    )
                                elif event["event"] == "delta":
                                    streamed_answer += event["data"].get("text", "")
                                    answer_placeholder.markdown(streamed_answer + "▌")
                                elif event["event"] == "done":
                                    result = event["data"]
                                elif event["event"] == "error":
                                    raise RuntimeError(event["data"].get("error", "流式回答失败"))
                        except Exception as exc:
                            progress_status.update(label="问答失败", state="error")
                            st.session_state.pop("pending_question", None)
                            st.error(f"问答失败：{exc}")
                        else:
                            st.session_state.pop("pending_question", None)
                            if result is None:
                                progress_status.update(label="问答失败", state="error")
                                st.error("问答失败：生成服务没有返回最终结果。")
                            else:
                                progress_status.update(label="回答完成", state="complete")
                                if resolution.is_followup and resolution.rewritten_query:
                                    result["conversation_rewritten_query"] = resolution.rewritten_query
                                answer_placeholder.markdown(result["answer"])
                                render_answer_details(result, paper_id=paper_id)
                                st.session_state.messages.append(
                                    {
                                        "role": "assistant",
                                        "content": result["answer"],
                                        "result": compact_result_for_history(result),
                                        "paper_id": paper_id,
                                    }
                                )
                                agent_memory.remember_result(
                                    result["answer"],
                                    result.get("sources", []),
                                    intent="qa",
                                    user_query=pending_question,
                                    rewritten_query=(
                                        resolution.rewritten_query if resolution.is_followup else None
                                    ),
                                )
                                persisted_result = compact_result_for_history(result)
                                persist_chat_exchange(
                                    user_content=pending_question,
                                    assistant_content=result["answer"],
                                    intent="qa",
                                    memory=agent_memory,
                                    rewritten_query=(
                                        resolution.rewritten_query if resolution.is_followup else None
                                    ),
                                    assistant_result=persisted_result,
                                    paper_id=paper_id,
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
        placeholder="提问、总结、对比、导出，或直接描述想查找的外部文献…",
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
        intent = ConversationResolver().resolve(clean_question, agent_memory.state).intent
        if intent == "literature_search":
            provider = choose_literature_source(clean_question)
            if not ensure_literature_source_ready(provider):
                return
        if intent == "qa" and multimodal_enabled:
            if not ensure_multimodal_ready():
                return
        elif intent in {"qa", "summary", "compare", "corpus_analysis"} and not has_papers:
            st.warning("当前没有已索引论文；如需外部检索，请明确说“查找/搜索论文”。")
            return
        requires_model = intent not in {
            "literature_search",
            "library_status",
            "source_explain",
            "export",
        }
        if not requires_model or ensure_api_ready():
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
    st.markdown('<div class="rag-sidebar-title">Conversations</div>', unsafe_allow_html=True)
    st.caption("管理会话，并指定本轮回答可使用的论文范围")

    with st.container(border=True):
        conversation_store = get_conversation_store()
        conversations = conversation_store.list_conversations(limit=50)
        conversation_ids = [item["id"] for item in conversations]
        conversation_by_id = {item["id"]: item for item in conversations}
        current_conversation_id = ensure_active_conversation()
        pending_picker = st.session_state.pop("pending_conversation_picker", None)
        if pending_picker in conversation_ids:
            st.session_state.conversation_picker = pending_picker
        elif st.session_state.get("conversation_picker") not in conversation_ids:
            st.session_state.conversation_picker = current_conversation_id

        selected_conversation_id = st.selectbox(
            "会话",
            conversation_ids,
            key="conversation_picker",
            format_func=lambda value: (
                f"{conversation_by_id[value]['title']} · "
                f"{conversation_by_id[value]['message_count']} 条消息"
            ),
            label_visibility="collapsed",
        )
        open_col, new_col = st.columns(2)
        if open_col.button(
            "打开",
            disabled=selected_conversation_id == current_conversation_id,
            width="stretch",
            key="open_conversation",
        ):
            activate_conversation(selected_conversation_id)
            st.session_state.pending_conversation_picker = selected_conversation_id
            st.rerun()
        if new_col.button("新建", width="stretch", key="new_conversation"):
            new_conversation_id = conversation_store.create_conversation()
            activate_conversation(new_conversation_id)
            st.session_state.pending_conversation_picker = new_conversation_id
            st.rerun()

    st.markdown('<div class="rag-sidebar-section">Retrieval scope</div>', unsafe_allow_html=True)
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
        with st.expander("多论文任务范围", expanded=False):
            compare_files = st.multiselect(
                "用于对比的论文",
                list(paper_options),
                default=[selected_file] if selected_file else [],
                key="workspace_compare_files",
            )
        selected_compare_ids = [paper_options[name] for name in compare_files]
        selected_pdf_path = PAPER_DIR / selected_file
        selected_chunks = next(
            (paper["chunk_count"] for paper in papers if paper["file_name"] == selected_file),
            0,
        )
        if selected_paper_id:
            single_deep = bool(st.session_state.get("single_paper_deep_enabled"))
            scope_summary = (
                f"单篇深度 Top-{settings.qa_detailed_top_k}"
                if single_deep
                else f"单篇 Top-{settings.retrieval_top_k}"
            )
        else:
            scope_summary = f"多篇深度 Top-{settings.multi_paper_top_k}"
        st.markdown(
            f'<div class="rag-file-status"><b>{selected_chunks}</b> chunks<br>'
            f'<span>{scope_summary}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        selected_file = None
        selected_paper_id = None
        selected_compare_ids = []
        selected_pdf_path = None
        st.warning("当前没有已索引论文。")

    with st.expander(f"论文集合 · {len(papers)} 篇", expanded=False):
        if papers:
            for paper in papers:
                active = paper["file_name"] == selected_file
                marker = "●" if active else "○"
                safe_file_name = html.escape(str(paper["file_name"]))
                st.markdown(
                    f'<div class="rag-file-row {"active" if active else ""}">'
                    f'<span>{marker}</span><div>{safe_file_name}'
                    f'<small>{paper["chunk_count"]} chunks</small></div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("上传后文件会显示在这里。")

    with st.expander("Quick upload", expanded=False):
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

    st.caption(f"本地 PDF：{len(local_pdfs)} · 已索引：{len(papers)}")
    st.page_link("pages/2_多模态论文阅读.py", label="多模态索引管理", width="stretch")

if st.session_state.pop("index_notice", None):
    st.success("论文索引已更新。")

if navigation == "Chat":
    agent_memory.set_scope(selected_paper_id, selected_compare_ids)
    active_document = html.escape(selected_file or "未选择论文")
    active_scope = "当前论文" if selected_paper_id else "全部论文"
    if st.session_state.get("multimodal_chat_enabled"):
        workspace_mode = "多模态已开启"
    elif selected_paper_id is None and papers:
        workspace_mode = "深度思考 · 完整规划"
    elif st.session_state.get("single_paper_deep_enabled"):
        workspace_mode = "深度思考"
    else:
        workspace_mode = "快速问答 · 章节级混合检索"
    st.markdown(
        f'<div class="rag-workspace-bar"><div class="rag-workspace-title">{active_document}</div>'
        f'<div class="rag-workspace-meta">{active_scope} · '
        f'{workspace_mode}</div></div>',
        unsafe_allow_html=True,
    )
    center, right = st.columns([1.55, 0.85], gap="medium")
    with center:
        render_chat_panel(selected_paper_id, bool(papers))
    with right:
        render_information_panel(
            selected_file=selected_file,
            selected_pdf_path=selected_pdf_path,
            memory=agent_memory,
        )

elif navigation == "Search":
    st.markdown(
        '<div class="rag-view-heading"><h2>智能文献检索</h2>'
        '<p>通过 IEEE API 或 arXiv MCP 查找外部论文，并保留完整工具调用记录</p></div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        with st.form("literature_search_form"):
            search_source = st.radio(
                "检索来源",
                ["IEEE Xplore", "arXiv MCP"],
                horizontal=True,
            )
            search_request = st.text_area(
                "检索需求",
                placeholder="例如：查找 2022 年以来关于多模态 RAG 的论文",
                height=110,
            )
            result_limit = st.slider("返回数量", min_value=5, max_value=25, value=10, step=5)
            submitted = st.form_submit_button(f"搜索 {search_source}", type="primary", width="stretch")
        if submitted:
            if not search_request.strip():
                st.warning("请输入文献检索需求。")
            else:
                provider = "arxiv" if search_source == "arXiv MCP" else "ieee"
                if not ensure_literature_source_ready(provider):
                    st.stop()
                try:
                    with st.spinner(f"正在生成检索式并查询 {search_source}..."):
                        service = (
                            get_arxiv_search_service()
                            if provider == "arxiv"
                            else get_literature_search_service()
                        )
                        st.session_state.literature_search_result = service.search(
                            search_request,
                            limit=result_limit,
                        )
                except Exception as exc:
                    st.error(f"文献检索失败：{exc}")

    if st.session_state.get("literature_search_result"):
        payload = st.session_state.literature_search_result
        formatter = (
            format_arxiv_search_answer
            if str(payload.get("source") or "").lower().startswith("arxiv")
            else format_search_answer
        )
        st.markdown(formatter(payload))
        render_literature_results(payload, key_prefix="search_page")
    elif not settings.ieee_is_ready:
        st.info("IEEE 需要配置 API Key；arXiv MCP 不需要 API Key，可直接切换使用。")

elif navigation == "Files":
    st.subheader("文件管理")
    outdated_papers = [paper for paper in papers if paper.get("index_outdated")]
    rebuildable_outdated = [
        (paper["file_name"], PAPER_DIR / paper["file_name"])
        for paper in outdated_papers
        if (PAPER_DIR / paper["file_name"]).exists()
    ]
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

        with st.container(border=True):
            st.markdown("#### 更新旧版索引")
            if outdated_papers:
                st.warning(
                    f"检测到 {len(outdated_papers)} 篇论文仍使用旧切分规则。"
                    "重建后会重新解析章节、切分并替换原索引。"
                )
            else:
                st.success("全部论文均使用当前切分规则。")
            if st.button(
                "重建全部旧版索引",
                disabled=not rebuildable_outdated,
                width="stretch",
                key="rebuild_outdated_indexes",
            ):
                if index_paths(rebuildable_outdated):
                    st.rerun()

    with library_col:
        metrics = st.columns(4)
        metrics[0].metric("已索引论文", len(papers))
        metrics[1].metric("向量 Chunks", vector_store.count_chunks())
        metrics[2].metric("本地 PDF", len(local_pdfs))
        metrics[3].metric("待重建", len(outdated_papers))
        st.caption(f"Collection：{settings.chroma_collection} · Embedding：{settings.embedding_model}")
        if papers:
            st.dataframe(
                [
                    {
                        "文件名": paper["file_name"],
                        "Paper ID": paper["paper_id"],
                        "Chunks": paper["chunk_count"],
                        "索引状态": "需重建" if paper.get("index_outdated") else "最新",
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

    tab_note, tab_compare, tab_debug = st.tabs(
        ["阅读笔记", "多论文对比与导出", "检索调试"]
    )
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

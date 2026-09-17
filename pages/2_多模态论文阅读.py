"""Streamlit page for multimodal paper reading with RAG-Anything."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import threading
from typing import Any

import streamlit as st

from backend.config import PAPER_DIR, RAG_ANYTHING_DIR, RAG_ANYTHING_OUTPUT_DIR, settings
from backend.rag_anything_reader import (
    RagAnythingPaperReader,
    check_rag_anything_dependencies,
    detect_compute_runtime,
)
from backend.ui import apply_app_style, render_page_header, render_top_navigation


st.set_page_config(page_title="多模态论文阅读", page_icon="MM", layout="wide")
apply_app_style()


class AsyncLoopRunner:
    """Run RAG-Anything coroutines on one persistent event loop."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, name="rag-anything-loop", daemon=True)
        self.thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()


@st.cache_resource
def get_async_runner() -> AsyncLoopRunner:
    return AsyncLoopRunner()


def run_async(coro):
    return get_async_runner().run(coro)


async def create_reader() -> RagAnythingPaperReader:
    try:
        from lightrag.kg.shared_storage import finalize_share_data

        finalize_share_data()
    except Exception:
        pass
    return RagAnythingPaperReader()


async def check_parser_ready(reader: RagAnythingPaperReader) -> bool:
    return reader.parser_ready()


def save_uploaded_document(uploaded_file) -> Path:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    file_path = PAPER_DIR / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())
    return file_path


@st.cache_resource
def get_reader() -> RagAnythingPaperReader:
    return run_async(create_reader())


def render_result(result: Any) -> None:
    if isinstance(result, (dict, list)):
        st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")
    elif result is None:
        st.info("任务已完成，但 RAG-Anything 没有返回额外结果。")
    else:
        st.markdown(str(result))


def multimodal_index_ready() -> bool:
    """Ignore cache-only files; require actual LightRAG index artifacts."""

    if not RAG_ANYTHING_DIR.exists():
        return False
    ignored = {".gitkeep", "kv_store_llm_response_cache.json"}
    return any(path.name not in ignored for path in RAG_ANYTHING_DIR.iterdir())


render_top_navigation("Multimodal", on_main_page=False)

render_page_header(
    "多模态论文阅读",
    "按需启动 RAG-Anything，解析论文文本、图片、表格和公式；页面浏览不会再加载整套解析器。",
    ["MinerU", "CUDA", "Vision LLM", "LightRAG"],
)

dependency_status = check_rag_anything_dependencies()
if not dependency_status.available:
    st.warning("当前环境还没有安装 RAG-Anything 依赖。")
    st.code(
        "& 'E:\\anaconda\\envs\\longchain\\python.exe' -m pip install \"raganything[all]\"",
        language="powershell",
    )
    st.write("安装后重新启动 Streamlit，再进入这个页面。Office 文档还需要单独安装 LibreOffice。")
    st.stop()

if not settings.is_ready:
    st.warning("请先在 .env 中配置 LLM_API_KEY、LLM_BASE_URL、LLM_MODEL 和 EMBEDDING_MODEL。")
    st.stop()

if not settings.vision_is_ready:
    st.warning("Please configure VISION_API_KEY, VISION_BASE_URL, and RAG_ANYTHING_VISION_MODEL in .env first.")
    st.stop()

runtime = detect_compute_runtime()
index_ready = multimodal_index_ready()
parsed_dirs = [path for path in RAG_ANYTHING_OUTPUT_DIR.iterdir() if path.is_dir()] if RAG_ANYTHING_OUTPUT_DIR.exists() else []

status_col1, status_col2, status_col3, status_col4 = st.columns(4)
status_col1.metric("计算设备", runtime.selected_device.upper())
status_col2.metric("CUDA", "可用" if runtime.cuda_available else "不可用")
status_col3.metric("GPU", runtime.gpu_name or "CPU")
status_col4.metric("多模态索引", "就绪" if index_ready else "尚未建立")

if runtime.cuda_available and runtime.vram_gb is not None and runtime.vram_gb <= 4.5:
    st.info(
        f"已识别 {runtime.gpu_name}（约 {runtime.vram_gb} GB 显存）。建议首次只解析 5～10 页；"
        "复杂公式和表格较多时，4 GB 显存可能触发回退或显存不足。"
    )

with st.sidebar:
    st.header("RAG-Anything 配置")
    st.write(f"工作目录：`{RAG_ANYTHING_DIR}`")
    st.write(f"解析输出：`{RAG_ANYTHING_OUTPUT_DIR}`")
    st.write(f"解析器：`{settings.rag_anything_parser}`")
    st.write(f"解析模式：`{settings.rag_anything_parse_method}`")
    st.write(f"解析后端：`{settings.rag_anything_backend or 'MinerU 默认'}`")
    st.write(f"默认设备：`{runtime.selected_device}`")
    st.write(f"PyTorch：`{runtime.torch_version}`")
    st.write(f"CUDA Runtime：`{runtime.cuda_version or '不可用'}`")
    st.write(f"视觉模型：`{settings.rag_anything_vision_model}`")
    st.write(f"Embedding 维度：`{settings.rag_anything_embedding_dim}`")
    st.write(f"解析输出目录：{len(parsed_dirs)} 个")
    st.caption("首次初始化 RAG-Anything 约需 60～90 秒；完成后会在当前进程内复用。")

    if st.button("检查解析器"):
        with st.spinner("首次检查会按需初始化 RAG-Anything..."):
            try:
                reader = get_reader()
                ready = run_async(check_parser_ready(reader))
            except Exception as exc:
                st.error(f"解析器初始化失败：{exc}")
            else:
                if ready:
                    st.success(f"解析器可用，将使用 {runtime.selected_device.upper()}。")
                else:
                    st.error("解析器不可用，请检查 MinerU/Docling/PaddleOCR 安装。")

left, right = st.columns([1, 1])

with left:
    st.subheader("建立多模态索引")
    uploaded_file = st.file_uploader(
        "上传论文或文档",
        type=["pdf", "doc", "docx", "ppt", "pptx", "xls", "xlsx", "jpg", "jpeg", "png", "bmp", "tiff", "webp"],
    )

    with st.expander("解析参数", expanded=False):
        lang = st.selectbox("语言", ["ch", "en", "ja", ""], index=0)
        device_options = ["cuda", "cpu"] if runtime.cuda_available else ["cpu"]
        device = st.selectbox("计算设备", device_options, index=0)
        use_page_range = st.checkbox("只解析部分页码")
        start_page = st.number_input("起始页码，0 表示第一页", min_value=0, value=0, disabled=not use_page_range)
        end_page = st.number_input("结束页码，0 表示第一页", min_value=0, value=10, disabled=not use_page_range)
        fast_mode = st.checkbox(
            "快速解析（关闭公式和表格模型）",
            value=False,
            help="适合先验证文本和图片流程；需要精确公式、表格时不要勾选。",
        )

    if st.button("解析并写入多模态索引", type="primary", disabled=uploaded_file is None):
        file_path = save_uploaded_document(uploaded_file)
        st.session_state["rag_anything_last_file"] = str(file_path)
        parse_status = st.status("正在初始化多模态解析器...", expanded=True)
        try:
            reader = get_reader()
            parse_status.write(
                f"计算设备：{device.upper()}；后端：{settings.rag_anything_backend or 'default'}；正在启动 MinerU。"
            )
            parse_status.write("正在提取文本、版面、图片、表格和公式，首次运行还会加载模型。")
            result = run_async(
                reader.process_document(
                    file_path,
                    start_page=int(start_page) if use_page_range else None,
                    end_page=int(end_page) if use_page_range else None,
                    lang=lang or None,
                    device=device,
                    formula=False if fast_mode else None,
                    table=False if fast_mode else None,
                )
            )
        except Exception as exc:
            parse_status.update(label="多模态解析失败", state="error", expanded=True)
            st.error(f"{type(exc).__name__}: {exc}")
        else:
            parse_status.update(label="多模态索引完成", state="complete", expanded=False)
            st.success(f"多模态索引完成：{file_path.name}")
            render_result(result)
            index_ready = multimodal_index_ready()

with right:
    st.subheader("多模态问答")
    question = st.text_area(
        "输入问题",
        value="请总结这篇论文的核心方法，并解释关键图表或公式传达了什么信息。",
        height=120,
    )
    mode = st.selectbox("检索模式", ["hybrid", "local", "global", "naive"], index=0)

    if not index_ready:
        st.warning("当前只有模型缓存或解析输出，没有可查询的 LightRAG 索引。请先成功完成一次多模态索引。")
    if st.button(
        "基于多模态索引提问",
        type="primary",
        disabled=not question.strip() or not index_ready,
        width="stretch",
    ):
        with st.spinner("正在进行混合检索和生成回答..."):
            try:
                result = run_async(get_reader().ask(question.strip(), mode=mode))
            except Exception as exc:
                st.error(f"多模态问答失败：{exc}")
            else:
                render_result(result)

st.divider()

with st.expander("直接多模态内容问答"):
    st.write("用于手动传入图片、表格或公式内容。图片的 img_path 必须是绝对路径。")
    content_json = st.text_area(
        "multimodal_content JSON",
        value='[\n  {\n    "type": "table",\n    "table_data": "| Method | Metric |\\n|---|---|\\n| Ours | 95.2% |",\n    "table_caption": "实验结果表"\n  }\n]',
        height=180,
    )
    direct_question = st.text_input("针对这些多模态内容提问", value="这个表格说明了什么？")

    if st.button("直接多模态问答", disabled=not direct_question.strip(), width="stretch"):
        try:
            multimodal_content = json.loads(content_json)
            if not isinstance(multimodal_content, list):
                raise ValueError("multimodal_content 必须是 JSON 数组。")
        except Exception as exc:
            st.error(f"JSON 解析失败：{exc}")
        else:
            with st.spinner("正在调用 RAG-Anything 多模态问答接口..."):
                try:
                    result = run_async(
                        get_reader().ask_with_content(
                            direct_question.strip(),
                            multimodal_content=multimodal_content,
                            mode=mode,
                        )
                    )
                except Exception as exc:
                    st.error(f"直接多模态问答失败：{exc}")
                else:
                    render_result(result)

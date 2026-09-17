"""Streamlit 多页面：观察 Chroma 向量数据库。"""

from __future__ import annotations

import streamlit as st

from backend.config import settings
from backend.embeddings import OpenAICompatibleClient
from backend.ui import apply_app_style, render_page_header, render_top_navigation
from backend.vector_store import ChromaVectorStore


st.set_page_config(page_title="向量数据库学习", page_icon="DB", layout="wide")
apply_app_style()

render_top_navigation("Tools", on_main_page=False)


@st.cache_resource
def get_vector_store() -> ChromaVectorStore:
    return ChromaVectorStore()


@st.cache_resource
def get_llm_client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient()


def render_chunk(row: dict, index: int) -> None:
    metadata = row.get("metadata", {})
    text = row.get("text", "")
    title = f"{index}. {metadata.get('file_name', 'unknown')} | 第 {metadata.get('page', '?')} 页"

    with st.expander(title, expanded=index == 1):
        st.code(metadata, language="python")
        st.write(text)


render_page_header(
    "RAG 向量数据库学习",
    "观察 Chroma 中的 chunk、章节 metadata 与混合检索排序结果。",
    ["Chroma", "Metadata", "Hybrid Retrieval"],
)

vector_store = get_vector_store()
papers = vector_store.list_papers()

left, right = st.columns([1, 2])

with left:
    st.subheader("数据库概览")
    st.metric("Collection", settings.chroma_collection)
    st.metric("总 chunk 数", vector_store.count_chunks())

    paper_options = {paper["file_name"]: paper["paper_id"] for paper in papers}
    selected_label = st.selectbox("选择论文", ["全部论文"] + list(paper_options.keys()))
    selected_paper_id = None if selected_label == "全部论文" else paper_options[selected_label]

    if selected_paper_id:
        st.metric("当前论文 chunk 数", vector_store.count_chunks(selected_paper_id))

    st.markdown("### 你需要观察什么")
    st.write("1. chunk 是否保留了页码。")
    st.write("2. metadata 是否包含 paper_id、file_name、page、chunk_id。")
    st.write("3. 检索结果的 distance 是否越小越相关。")
    st.write("4. top_k 太小会漏信息，太大会塞入噪声。")

with right:
    st.subheader("查看数据库中的 chunk")
    preview_limit = st.slider("查看条数", min_value=1, max_value=20, value=5)
    rows = vector_store.peek_chunks(limit=preview_limit, paper_id=selected_paper_id)

    if not rows:
        st.info("还没有 chunk。请先回到主页上传 PDF 并建立索引。")
    else:
        for index, row in enumerate(rows, start=1):
            render_chunk(row, index)

st.divider()

st.subheader("手动测试语义检索")
st.write("这里不会调用大模型回答，只展示向量数据库找到了哪些 chunk。")

query_text = st.text_input("输入检索问题", value="这篇论文的核心方法是什么？")
top_k = st.slider("top_k", min_value=1, max_value=10, value=settings.retrieval_top_k)

if st.button("执行向量检索", type="primary"):
    if not settings.is_ready:
        st.warning("请先配置 .env 中的模型 API 信息，因为检索问题本身也需要 embedding。")
    else:
        with st.spinner("正在把问题向量化并检索 Chroma..."):
            hits = vector_store.query(
                query_text=query_text,
                embedding_client=get_llm_client(),
                top_k=top_k,
                paper_id=selected_paper_id,
            )

        if not hits:
            st.info("没有检索到结果。")
        else:
            st.markdown("### 检索结果")
            for index, hit in enumerate(hits, start=1):
                explained = vector_store.explain_hit(hit)
                distance = explained["distance"]
                distance_text = f"{distance:.4f}" if isinstance(distance, float) else str(distance)
                with st.expander(
                    f"Top {index} | distance={distance_text} | 第 {explained['page']} 页",
                    expanded=index == 1,
                ):
                    st.code(
                        {
                            "paper_id": explained["paper_id"],
                            "file_name": explained["file_name"],
                            "page": explained["page"],
                            "chunk_id": explained["chunk_id"],
                            "distance": explained["distance"],
                        },
                        language="python",
                    )
                    st.write(explained["text"])

"""论文总结模块。"""

from __future__ import annotations

from typing import Any

from backend.config import settings
from backend.embeddings import OpenAICompatibleClient
from backend.prompts import build_summary_messages
from backend.vector_store import ChromaVectorStore


SUMMARY_QUERY = "研究背景 研究问题 核心方法 创新点 实验设置 实验结果 优点 缺点 可改进方向"


def generate_paper_note(
    paper_id: str,
    vector_store: ChromaVectorStore,
    llm_client: OpenAICompatibleClient,
    top_k: int = 12,
) -> dict[str, Any]:
    """基于检索到的代表性片段生成论文阅读笔记。"""

    hits = vector_store.query(
        query_text=SUMMARY_QUERY,
        embedding_client=llm_client,
        top_k=max(top_k, settings.retrieval_top_k),
        paper_id=paper_id,
    )
    if not hits:
        return {"note": "论文中没有找到明确依据。", "sources": []}

    messages = build_summary_messages(hits)
    note = llm_client.chat_long_form(
        messages,
        temperature=0.1,
        max_tokens=settings.summary_max_tokens,
        model_candidates=settings.qa_long_models,
        enable_thinking=settings.qa_long_enable_thinking,
        continuation_instruction=(
            "上一段论文阅读卡片因长度限制中断。请从中断处继续，补完尚未完成的栏目，"
            "不要重复已写内容，继续沿用原来源编号。只输出续写正文。"
        ),
    )
    return {"note": note, "sources": hits}

"""多篇论文对比模块：先基于阅读卡片对比，避免全文混检。"""

from __future__ import annotations

from backend.config import settings
from backend.embeddings import OpenAICompatibleClient
from backend.prompts import build_compare_messages


def compare_paper_cards(cards: list[str], llm_client: OpenAICompatibleClient) -> str:
    """输入多篇论文阅读卡片，输出 Markdown 对比表。"""

    if len(cards) < 2:
        return "请至少提供两篇论文阅读卡片后再进行对比。"
    messages = build_compare_messages(cards)
    return llm_client.chat_long_form(
        messages,
        temperature=0.1,
        max_tokens=settings.compare_max_tokens,
        model_candidates=settings.qa_long_models,
        enable_thinking=settings.qa_long_enable_thinking,
        continuation_instruction=(
            "上一段多论文对比报告因长度限制中断。请从中断处继续，补完剩余论文、对比矩阵、"
            "机制差异、选择建议和研究空间，不要重复已写内容。只输出续写正文。"
        ),
    )

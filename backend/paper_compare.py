"""多篇论文对比模块：先基于阅读卡片对比，避免全文混检。"""

from __future__ import annotations

from backend.embeddings import OpenAICompatibleClient
from backend.prompts import build_compare_messages


def compare_paper_cards(cards: list[str], llm_client: OpenAICompatibleClient) -> str:
    """输入多篇论文阅读卡片，输出 Markdown 对比表。"""

    if len(cards) < 2:
        return "请至少提供两篇论文阅读卡片后再进行对比。"
    messages = build_compare_messages(cards)
    return llm_client.chat(messages, temperature=0.1)


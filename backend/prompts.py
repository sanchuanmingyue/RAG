"""Prompt templates for paper reading tasks."""

from __future__ import annotations

import re
from typing import Any

from backend.config import settings


QA_SYSTEM_PROMPT = """你是一个严谨的论文阅读助手。你只能根据给定的论文片段回答问题。

规则：
1. 每个关键结论后必须标注来源编号，例如 [S1]、[S2]。
2. 证据可以是与问题含义一致的直接陈述、同义表达、定义、公式或可由片段直接推出的结论；不要求与问题逐字相同。
3. 只要任一来源包含直接或同义证据，就必须回答，并引用实际支持答案的来源。
4. 主题相关不等于能够回答。来源必须直接支持问题所问的实体、关系、数字或结论；不要用邻近的其他实验、方法、数据集或结果代替答案。
5. 不要因为来源同时包含无关内容而忽略其中的有效证据，也不要编造片段中没有的信息。
6. 只能引用实际提供的来源编号，禁止引用不存在的编号。
7. 优先给出简洁、可追溯的回答。普通事实题控制在 2～5 句话，列举题最多列出 5 项。
8. 不要输出分析过程、自我复核过程、草稿或“让我重新阅读”等内部推理；只输出最终答案和引用。"""


SUMMARY_SYSTEM_PROMPT = """你是一个严谨的论文阅读助手。请只根据给定论文片段生成论文阅读笔记。如果某一项没有明确依据，请写“论文中没有找到明确依据”。"""


COMPARE_SYSTEM_PROMPT = """你是一个文献综述助手。请基于给定的多篇论文阅读卡片做对比，不要编造卡片中没有的信息。"""


def build_context(chunks: list[dict[str, Any]]) -> str:
    """Build bounded context with multi-chunk evidence for each ranked section."""

    lines: list[str] = []
    remaining_chars = max(settings.qa_context_max_chars, len(chunks))
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        page = metadata.get("page", "?")
        parent_pages = metadata.get("parent_pages")
        page_text = f"pages {parent_pages}" if parent_pages else f"page {page}"
        file_name = metadata.get("file_name", "unknown")
        section_path = metadata.get("section_path") or "unknown section"
        section_type = metadata.get("section_type") or "unknown"
        source_id = f"S{index}"
        candidate_text = str(
            chunk.get("generation_text")
            or chunk.get("section_document")
            or chunk.get("text")
            or ""
        ).strip()
        remaining_sources = len(chunks) - index + 1
        fair_share = max(remaining_chars // remaining_sources, 1)
        source_budget = min(max(settings.qa_source_max_chars, 1), fair_share)
        source_text = candidate_text[:source_budget]
        remaining_chars = max(remaining_chars - len(source_text), 0)
        lines.append(
            f"[{source_id} | file: {file_name} | {page_text} | "
            f"section: {section_path} | section_type: {section_type}]\n{source_text}"
        )
    return "\n\n".join(lines)


def is_yes_no_question(question: str) -> bool:
    """Identify binary questions without imposing the format on open questions."""

    normalized = " ".join(question.strip().split()).lower()
    if not normalized:
        return False
    if any(marker in normalized for marker in ("是否", "能否", "可否", "是不是", "有没有", "需不需要", "会不会")):
        return True
    if normalized.endswith(("吗？", "吗?", "吗", "么？", "么?")):
        return True
    return bool(
        re.match(
            r"^(?:is|are|was|were|do|does|did|can|could|will|would|should|has|have|had|"
            r"must|may|might)\b",
            normalized,
        )
    )


def answer_language(question: str) -> str:
    """Use Chinese only when the question itself contains Chinese text."""

    return "zh" if re.search(r"[\u4e00-\u9fff]", question) else "en"


def is_multi_paper_question(question: str) -> bool:
    """Detect questions that explicitly require evidence across papers."""

    normalized = " ".join(question.lower().split())
    return bool(
        re.search(r"(?:多篇|各篇|每篇|这些|所有|全部).{0,8}(?:论文|文章|文献|研究)", normalized)
        or re.search(r"(?:论文|文章|文献|研究).{0,8}(?:分别|共同|综合|整体|之间)", normalized)
        or re.search(r"(?:他们|它们|这些|这几篇).{0,8}(?:分别|各自|共同|之间)", normalized)
        or re.search(r"\b(?:papers|articles|studies|publications)\b", normalized)
    )


def build_qa_messages(question: str, chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    context = build_context(chunks)
    language_instruction = (
        "请严格使用中文回答。若证据不能直接且无歧义地回答问题，只输出“论文中没有找到明确依据。”。"
        if answer_language(question) == "zh"
        else "Answer strictly in English. If the evidence does not directly and unambiguously answer the question, output only: "
        '"I could not find explicit evidence in the paper."'
    )
    binary_instruction = ""
    if is_yes_no_question(question):
        binary_instruction = (
            "\n这是一个 Yes/No（是否）问题。请先判断问题中的核心谓词被证据肯定还是否定，"
            "第一句必须以“是，”或“否，”明确作答；“解决需要 X 的问题”不等于“仍然需要 X”。"
            "输出前必须检查首句极性与后文文字及公式一致；A≥B 与 B≤A 表达同一关系。"
            if answer_language(question) == "zh"
            else "\nThis is a Yes/No question. Start the first sentence with 'Yes,' or 'No,' and verify "
            "that this polarity agrees with the explanation and any inequalities that follow."
        )
    multi_paper_instruction = ""
    if is_multi_paper_question(question):
        paper_names = list(
            dict.fromkeys(
                str(chunk.get("metadata", {}).get("file_name") or "unknown")
                for chunk in chunks
            )
        )
        paper_list = "、".join(paper_names)
        multi_paper_instruction = (
            "\n这是多论文综合问题。请按论文分别归纳，再总结共同点或差异；每篇论文的结论应引用该论文自己的来源。"
            f"本次证据来自 {len(paper_names)} 篇论文：{paper_list}。"
            "同一 file 的多个来源编号属于同一篇论文；必须以 file 字段识别论文，不能把 S1、S2 等来源编号当作不同论文。"
            "每个不同的 file 最多生成一个论文小节，小节标题优先使用文件名。"
            "如果证据只覆盖部分论文，回答有依据的部分并明确哪些论文缺少证据，不要因此拒绝整个问题。"
            if answer_language(question) == "zh"
            else "\nThis is a multi-paper synthesis question. Summarize each paper separately before identifying "
            f"shared themes or differences. The evidence covers {len(paper_names)} papers: {', '.join(paper_names)}. "
            "Multiple source IDs with the same file field belong to one paper; never treat S1, S2, and other source "
            "IDs as separate papers. Create at most one section per distinct file and cite evidence from that paper. "
            "If evidence covers only "
            "some papers, answer for those papers and identify the uncovered papers instead of refusing the whole question."
        )
    user_prompt = f"""论文片段：
{context}

用户问题：
{question}

请先逐一检查所有来源是否包含直接证据、同义表达、定义、公式或可直接推出的结论。
只要存在一项有效证据就回答，并在对应结论后引用来源编号，例如 [S1]。
仅仅主题相似、出现相同关键词或描述相邻任务不构成有效证据。问题指代不明确、存在多个可能对象，
或关键数字/名称被占位符替代时，不要猜测，必须拒答。
本次唯一合法的来源编号范围是 [S1] 到 [S{len(chunks)}]，不要生成范围外的编号。
除非问题明确要求详细分析，否则直接回答问题，不要复述检索过程或解释你如何作出判断。
{language_instruction}{binary_instruction}{multi_paper_instruction}"""
    return [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_summary_messages(chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    context = build_context(chunks)
    user_prompt = f"""论文片段：
{context}

请生成一份 Markdown 格式的论文阅读笔记，包含以下栏目：
- 研究背景
- 研究问题
- 核心方法
- 创新点
- 实验设置
- 实验结果
- 优点
- 缺点
- 可改进方向

每一项都尽量附上来源编号。"""
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_compare_messages(cards: list[str]) -> list[dict[str, str]]:
    joined_cards = "\n\n---\n\n".join(cards)
    user_prompt = f"""论文阅读卡片：
{joined_cards}

请生成一个 Markdown 对比表，列包括：论文、核心方法、数据集/实验设置、主要创新、优势、局限、可改进方向。表格后再用 3-5 条总结这些论文的共同点和差异。"""
    return [
        {"role": "system", "content": COMPARE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

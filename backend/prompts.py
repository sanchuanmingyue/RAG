"""Prompt templates for paper reading tasks."""

from __future__ import annotations

import re
from typing import Any

from backend.config import settings


QA_SYSTEM_PROMPT = """你是一个严谨的论文阅读助手。你只能根据给定的论文片段回答问题。

规则：
1. 你的首要任务是直接回答用户所问的内容，而不是报告“是否存在依据”。用户问“是什么、有哪些、如何、请介绍或请总结”时，必须给出具体事实、过程、数字或结论；不得只回答“有依据”“有证据”或类似元话语。
2. 在内部检查证据是否充分，不要把检查过程或检查结论当成最终答案。
3. 每个关键结论后必须标注来源编号，例如 [S1]、[S2]。
4. 证据可以是与问题含义一致的直接陈述、同义表达、定义、公式或可由片段直接推出的结论；不要求与问题逐字相同。
5. 只要任一来源包含直接或同义证据，就必须回答有证据支持的部分，并引用实际支持答案的来源。
6. 主题相关不等于能够回答。来源必须直接支持问题所问的实体、关系、数字或结论；不要用邻近的其他实验、方法、数据集或结果代替答案。
7. 不要因为来源同时包含无关内容而忽略其中的有效证据，也不要编造片段中没有的信息。
8. 只能引用实际提供的来源编号，禁止引用不存在的编号。
9. 普通事实题控制在 2～5 句话；用户明确要求详细介绍时，应按主题分点组织并覆盖来源中能确认的主要内容。
10. 不要输出分析过程、自我复核过程、草稿或“让我重新阅读”等内部推理；只输出最终答案和引用。"""


SUMMARY_SYSTEM_PROMPT = """你是一个严谨的论文阅读助手。请只根据给定论文片段生成结构化阅读笔记。
除了抽取事实，还要解释论文从研究动机、问题建模到方法设计和实验验证的逻辑链；不得补充来源之外的信息。如果某一项没有明确依据，请写“论文中没有找到明确依据”。"""


COMPARE_SYSTEM_PROMPT = """你是一个严谨的文献综述助手。请基于给定的多篇论文阅读卡片做深度对比，不要编造卡片中没有的信息。
你的任务不是罗列摘要，而是解释每篇论文解决什么问题、为什么采用该设计、方法模块如何关联、实验怎样支撑结论，以及论文之间真正可比的共同点和差异。"""


def build_context(chunks: list[dict[str, Any]], *, long_form: bool = False) -> str:
    """Build bounded context with multi-chunk evidence for each ranked section."""

    lines: list[str] = []
    context_limit = (
        settings.qa_long_context_max_chars
        if long_form
        else settings.qa_context_max_chars
    )
    per_source_limit = (
        settings.qa_long_source_max_chars
        if long_form
        else settings.qa_source_max_chars
    )
    remaining_chars = max(context_limit, len(chunks))
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
        source_budget = min(max(per_source_limit, 1), fair_share)
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


def is_long_form_question(question: str) -> bool:
    """Identify requests that need broader evidence and a structured synthesis."""

    normalized = " ".join(question.lower().split())
    if is_multi_paper_question(normalized):
        return True
    markers = (
        "详细", "全面", "深入", "系统介绍", "系统梳理", "逐章", "逐节",
        "完整介绍", "完整分析", "分别在做什么", "对比", "比较", "差异",
        "共同点", "优缺点", "局限", "为什么", "如何设计", "设计思路",
        "in detail", "detailed", "comprehensive", "step by step", "compare",
        "comparison", "why", "deep analysis",
    )
    if any(marker in normalized for marker in markers):
        return True
    # Long compound questions usually contain several requested dimensions.
    separators = sum(normalized.count(marker) for marker in ("，", "、", "以及", "并且", " and "))
    return len(normalized) >= 80 and separators >= 2


def build_evidence_plan_messages(
    question: str,
    chunks: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build a private evidence-outline request for complex paper questions."""

    context = build_context(chunks, long_form=True)
    language = "中文" if answer_language(question) == "zh" else "English"
    prompt = f"""你正在为一篇长篇、可核验的论文回答准备证据提纲。不要直接写最终答案，也不要输出思维过程。

论文来源：
{context}

用户问题：
{question}

请用{language}输出一份简洁的“写作证据提纲”：
1. 先识别用户实际要求回答的维度；
2. 多论文问题必须按 file 分组，保证每篇论文都独立归纳；
3. 对每个维度列出可确认的事实、机制、变量关系、实验数字及对应 [S编号]；
4. 标出论文之间可直接比较的共同点、差异和因果联系；
5. 明确证据空缺，禁止猜测。

只输出事实型提纲，不写最终开场、结尾或检索过程。"""
    return [
        {
            "role": "system",
            "content": "你是论文证据规划器。只整理来源支持的事实与结构，不补充来源外知识。",
        },
        {"role": "user", "content": prompt},
    ]


def build_qa_messages(
    question: str,
    chunks: list[dict[str, Any]],
    *,
    evidence_plan: str | None = None,
    long_form: bool | None = None,
) -> list[dict[str, str]]:
    resolved_long_form = is_long_form_question(question) if long_form is None else long_form
    context = build_context(chunks, long_form=resolved_long_form)
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
    open_question_instruction = ""
    if not is_yes_no_question(question):
        open_question_instruction = (
            "\n这是开放式内容问题。第一句就给出问题所要求的具体内容，不要以“是/否”开头，"
            "也不要把“论文中有明确依据”“来源支持该结论”当作答案。"
            if answer_language(question) == "zh"
            else "\nThis is an open content question. Begin with the requested facts or explanation, not Yes/No or a statement that evidence exists."
        )
    detail_instruction = ""
    normalized_question = question.lower()
    if any(marker in normalized_question for marker in ("详细", "全面", "系统介绍", "深入", "in detail", "detailed")):
        detail_instruction = (
            "\n用户明确要求详细回答。请使用简短小标题或项目符号组织内容；信息缺失的项目直接省略，"
            "不要因为某一细节缺失而拒绝整个问题。"
            if answer_language(question) == "zh"
            else "\nThe user explicitly requests detail. Organize the supported content with short headings or bullets; omit unsupported fields instead of refusing the whole question."
        )
    long_form_instruction = ""
    if resolved_long_form:
        long_form_instruction = (
            "\n这是深度问答。先给出一段能概括全文逻辑的总体主线，再按问题涉及的论文、章节或维度展开。"
            "每个重要部分应尽量回答：研究对象是什么、为什么这样设计、具体如何实现、关键变量或模块如何关联、"
            "实验如何验证以及能够得到什么结论。对论文事实、作者解释和基于多处证据的综合判断要明确区分。"
            "不要为了简短而省略来源中已有的关键机制、参数、数字、反例或局限，也不要重复同一句结论来凑长度。"
            "当来源覆盖多个章节或多个比较维度时，应逐项完成内部证据提纲后再结束；中文回答通常应充分展开到"
            "约 1500～3000 字，但以证据覆盖度为准，不得用空话凑字数。"
            if answer_language(question) == "zh"
            else "\nThis is a deep-answer request. Start with the paper's overall logic, then expand by paper, section, or requested dimension. Explain what is done, why it is designed that way, how it works, how variables or modules relate, how experiments validate it, and what follows from the evidence. Distinguish paper facts, author explanations, and cross-source synthesis."
        )
    experiment_instruction = ""
    if any(
        marker in normalized_question
        for marker in ("实验", "结果", "评估", "消融", "基线", "experiment", "result", "evaluation", "ablation", "baseline")
    ):
        experiment_instruction = (
            "\n这是实验相关问题。优先提取来源中实际出现的实验环境或参数、对比方法、评价指标、"
            "定量结果和作者据此得到的结论。必须写出具体内容，不能只说实验有效或存在证据。"
            if answer_language(question) == "zh"
            else "\nThis is an experiment question. Extract concrete setup or parameters, baselines, metrics, quantitative results, and supported conclusions; never answer only that experiments or evidence exist."
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
    evidence_plan_section = ""
    if evidence_plan:
        evidence_plan_section = f"""

内部证据提纲（用于保证覆盖完整；仍须逐项以原始来源复核，不要向用户提及该提纲）：
{evidence_plan}
"""
    user_prompt = f"""论文片段：
{context}
{evidence_plan_section}

用户问题：
{question}

请在内部逐一检查所有来源是否包含直接证据、同义表达、定义、公式或可直接推出的结论，不要输出检查过程。
只要存在一项有效证据就回答，并在对应结论后引用来源编号，例如 [S1]。
仅仅主题相似、出现相同关键词或描述相邻任务不构成有效证据。问题指代不明确、存在多个可能对象，
或关键数字/名称被占位符替代时，不要猜测，必须拒答。
本次唯一合法的来源编号范围是 [S1] 到 [S{len(chunks)}]，不要生成范围外的编号。
除非问题明确要求详细分析，否则直接回答问题，不要复述检索过程或解释你如何作出判断。
    {language_instruction}{binary_instruction}{open_question_instruction}{detail_instruction}{long_form_instruction}{experiment_instruction}{multi_paper_instruction}"""
    return [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_summary_messages(chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    context = build_context(chunks, long_form=True)
    user_prompt = f"""论文片段：
{context}

请生成一份 Markdown 格式的论文阅读笔记，包含以下栏目：
- 全文逻辑主线（为什么做 → 如何建模 → 为什么采用该方法 → 如何验证）
- 研究背景
- 研究问题
- 系统模型与关键变量
- 核心方法与算法流程
- 设计动机（解释关键模块为什么这样设计）
- 创新点
- 实验设置
- 实验结果
- 优点
- 缺点
- 可改进方向

每一项都尽量附上来源编号。保留来源中的重要公式含义、参数和定量结果，避免只写概括性结论。"""
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_compare_messages(cards: list[str]) -> list[dict[str, str]]:
    joined_cards = "\n\n---\n\n".join(cards)
    user_prompt = f"""论文阅读卡片：
{joined_cards}

请生成一份结构化的深度对比报告：

1. **总体主线**：先用一段话解释这些论文分别处在什么研究问题上，以及它们之间的关系。
2. **逐篇解析**：每篇论文单独成节，说明研究问题、设计动机、核心机制/算法流程、优化目标、实验设置、关键结果与局限。不要只改写卡片栏目。
3. **横向对比矩阵**：至少比较研究对象、系统模型、决策变量、核心算法、时间尺度、数据/实验环境、评价指标、主要收益和局限；不适用或证据缺失时明确写出。
4. **机制层面的共同点与差异**：解释为什么方法不同、各自牺牲了什么、适合什么条件，不要只列关键词。
5. **选择建议与可研究空间**：基于卡片证据说明不同研究目标下更适合参考哪篇，并归纳尚未解决的问题。

回答应充分展开，保留卡片中出现的重要变量、参数和定量结果；避免重复和空泛评价。"""
    return [
        {"role": "system", "content": COMPARE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

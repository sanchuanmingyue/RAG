"""RAG question answering: analyze query, retrieve, and answer with citations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Iterator
from typing import Any

from backend.config import settings
from backend.embeddings import OpenAICompatibleClient
from backend.prompts import answer_language, build_qa_messages, is_multi_paper_question, is_yes_no_question
from backend.vector_store import ChromaVectorStore


REFUSAL = "论文中没有找到明确依据。"
ENGLISH_REFUSAL = "I could not find explicit evidence in the paper."


def refusal_for_question(question: str) -> str:
    return REFUSAL if answer_language(question) == "zh" else ENGLISH_REFUSAL

_REFUSAL_PATTERNS = (
    re.compile(r"论文中(?:没有|未)(?:找到)?明确依据"),
    re.compile(r"(?:给定|提供的)?(?:论文)?片段(?:中)?(?:没有|未)(?:提供|提及|包含).{0,120}(?:依据|信息|内容|答案)"),
    re.compile(r"(?:无法|不能)(?:仅)?(?:根据|据此|从)(?:给定|提供的|这些)?(?:论文)?片段.{0,30}(?:回答|确定|确认|判断)"),
    re.compile(r"(?:根据|基于)(?:给定|提供的|这些)?(?:论文)?片段.{0,80}(?:无法|不能)(?:回答|确定|确认|判断)"),
    re.compile(r"(?:未说明|未提及|没有说明|没有提及).{0,100}(?:具体|目标|答案|内容|信息)?"),
    re.compile(r"与.{0,60}(?:问题|目标).{0,30}无关"),
    re.compile(r"(?:证据|依据|信息)(?:不足|不充分).{0,20}(?:回答|确定|确认|判断)?"),
    re.compile(r"(?:insufficient|not enough|no) (?:evidence|information).{0,40}(?:answer|determine|conclude)?", re.I),
    re.compile(r"(?:could not|couldn't|cannot|can't) find (?:explicit )?(?:evidence|information)", re.I),
    re.compile(r"(?:cannot|can't|unable to) (?:answer|determine|conclude).{0,40}(?:provided|given|sources|context)", re.I),
)


@dataclass
class QueryAnalysis:
    original_question: str
    rewritten_query: str
    intent: str
    section_types: list[str] | None


def infer_section_types(question: str) -> list[str] | None:
    analysis = analyze_question(question)
    return analysis.section_types


def analyze_question(question: str) -> QueryAnalysis:
    """Rule-based query intent classification and retrieval query rewriting."""

    normalized = question.lower().strip()

    term_groups: list[tuple[str, tuple[str, ...], list[str], list[str]]] = [
        (
            "background",
            (
                "research background",
                "problem background",
                "motivation",
                "研究背景",
                "问题背景",
                "研究动机",
                "背景",
            ),
            ["abstract", "introduction", "related_work"],
            ["research background", "problem context", "motivation", "research gap"],
        ),
        (
            "experiment",
            (
                "experiment",
                "experimental",
                "baseline",
                "dataset",
                "data set",
                "benchmark",
                "evaluation",
                "evaluate",
                "metric",
                "ablation",
                "result",
                "performance",
                "accuracy",
                "实验",
                "基线",
                "数据集",
                "评估",
                "评价",
                "指标",
                "消融",
                "结果",
                "性能",
                "准确率",
                "对比",
            ),
            ["experiment", "result"],
            ["experimental setup", "dataset", "baseline", "evaluation metric", "ablation", "results"],
        ),
        (
            "method",
            (
                "method",
                "approach",
                "model",
                "algorithm",
                "architecture",
                "framework",
                "training",
                "optimization",
                "方法",
                "模型",
                "算法",
                "架构",
                "框架",
                "训练",
                "优化",
            ),
            ["method"],
            ["method", "approach", "model architecture", "algorithm", "framework"],
        ),
        (
            "related_work",
            (
                "related work",
                "prior work",
                "previous work",
                "literature",
                "相关工作",
                "相关研究",
                "已有工作",
                "现有工作",
                "文献",
            ),
            ["related_work"],
            ["related work", "prior work", "literature review", "background"],
        ),
        (
            "contribution",
            (
                "contribution",
                "innovation",
                "novelty",
                "novel",
                "main idea",
                "propose",
                "contribute",
                "创新",
                "贡献",
                "亮点",
                "新颖",
                "提出",
                "核心思想",
            ),
            ["abstract", "introduction", "method", "conclusion"],
            ["contribution", "novelty", "main idea", "proposed method", "conclusion"],
        ),
    ]

    for intent, terms, section_types, expansions in term_groups:
        if any(term in normalized for term in terms):
            return QueryAnalysis(
                original_question=question,
                rewritten_query=_rewrite_query(question, expansions),
                intent=intent,
                section_types=section_types,
            )

    return QueryAnalysis(
        original_question=question,
        rewritten_query=_rewrite_query(question, []),
        intent="general",
        section_types=None,
    )


def _rewrite_query(question: str, expansions: list[str]) -> str:
    """Lightweight rewrite: keep the user question and append retrieval hints."""

    cleaned = re.sub(r"\s+", " ", question).strip()
    if not expansions:
        return cleaned
    expansion_text = " ".join(expansions)
    return f"{cleaned} {expansion_text}"


def _has_citation(answer: str) -> bool:
    return bool(
        extract_source_citation_ids(answer)
        or re.search(r"\[第\s*\d+\s*页\]", answer)
        or re.search(r"\[page\s*\d+\]", answer, flags=re.I)
    )


_SOURCE_CITATION_BLOCK_PATTERN = re.compile(r"\[(?=[^\]]*\bS\s*\d)[^\]]+\]", re.I)


def extract_source_citation_ids(answer: str) -> list[int]:
    """Extract source IDs from simple, grouped, and ranged citation blocks."""

    cited_ids: list[int] = []
    for block_match in _SOURCE_CITATION_BLOCK_PATTERN.finditer(answer):
        block = block_match.group(0)
        for source_match in re.finditer(
            r"\bS\s*(\d+)(?:\s*[-–—]\s*S?\s*(\d+))?\b",
            block,
            re.I,
        ):
            start = int(source_match.group(1))
            end_text = source_match.group(2)
            if end_text is None:
                cited_ids.append(start)
                continue
            end = int(end_text)
            step = 1 if end >= start else -1
            cited_ids.extend(range(start, end + step, step))
    return cited_ids


def _invalid_source_citations(answer: str, source_count: int) -> list[int]:
    return sorted({source_id for source_id in extract_source_citation_ids(answer) if not 1 <= source_id <= source_count})


def repair_source_citations(answer: str, source_count: int) -> tuple[str, list[int]]:
    """Remove out-of-range IDs while preserving valid citations in each block."""

    invalid_ids = _invalid_source_citations(answer, source_count)
    if not invalid_ids:
        return answer, []

    def replace_block(match: re.Match[str]) -> str:
        valid_ids: list[int] = []
        for source_id in extract_source_citation_ids(match.group(0)):
            if 1 <= source_id <= source_count and source_id not in valid_ids:
                valid_ids.append(source_id)
        return "" if not valid_ids else "[" + ", ".join(f"S{source_id}" for source_id in valid_ids) + "]"

    repaired = _SOURCE_CITATION_BLOCK_PATTERN.sub(replace_block, answer)
    repaired = re.sub(r"[ \t]+([，。,.!?；;：:])", r"\1", repaired)
    return repaired.strip(), invalid_ids


def _needs_binary_consistency_retry(question: str, answer: str) -> bool:
    if not settings.binary_consistency_retry or not is_yes_no_question(question):
        return False
    normalized = " ".join(answer.lower().split())
    negative_lead = bool(re.match(r"^(?:否|不|no\b|false\b)", normalized, re.I))
    comparison_explanation = bool(
        re.search(r"(?:相反|反而|however|but).{0,160}(?:<=|>=|≤|≥|小于或等于|大于或等于)", normalized, re.I)
    )
    return negative_lead and comparison_explanation


def _citation_retry_messages(
    messages: list[dict[str, str]], raw_answer: str, source_count: int
) -> list[dict[str, str]]:
    """Ask once for the same supported answer with valid source citations."""

    return messages + [
        {"role": "assistant", "content": raw_answer},
        {
            "role": "user",
            "content": (
                f"上面的回答包含实质内容，但缺少可核验引用。请保留有论文证据支持的内容，"
                f"并在每个关键结论后补充对应来源编号；唯一合法范围是 [S1] 到 [S{source_count}]。"
                "不要新增原回答和来源中没有的事实，只输出修正后的完整回答。"
            ),
        },
    ]


def classify_refusal_answer(answer: str) -> str | None:
    """Classify a model response as a pure or mixed refusal.

    A mixed refusal contains both a refusal sentence and a substantive answer.
    It remains answerable for evaluation but is surfaced as a warning.
    """

    normalized = re.sub(r"\s+", " ", answer).strip()
    if not normalized:
        return "pure_refusal"
    sentences = [part.strip() for part in re.split(r"(?<=[。！？.!?])\s+|[\r\n]+", answer) if part.strip()]
    refusal_sentences = [
        sentence for sentence in sentences if any(pattern.search(sentence) for pattern in _REFUSAL_PATTERNS)
    ]
    if not refusal_sentences:
        return None
    substantive_sentences = [sentence for sentence in sentences if sentence not in refusal_sentences]
    substantive_size = sum(
        len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", re.sub(r"\[\s*S\s*\d+\s*\]", "", sentence, flags=re.I)))
        for sentence in substantive_sentences
    )
    return "mixed_refusal" if substantive_size >= 12 else "pure_refusal"


def is_refusal_answer(answer: str) -> bool:
    """Return true only when the response is a pure refusal."""

    return classify_refusal_answer(answer) == "pure_refusal"


def _strip_trailing_refusal(answer: str) -> str:
    """Remove standalone trailing refusal paragraphs while preserving raw output."""

    paragraphs = [part.strip() for part in re.split(r"\r?\n\s*\r?\n", answer.strip()) if part.strip()]
    while len(paragraphs) > 1 and classify_refusal_answer(paragraphs[-1]) == "pure_refusal":
        paragraphs.pop()
    return "\n\n".join(paragraphs).strip()


def _diversify_hits_by_paper(hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Round-robin ranked hits so multi-paper synthesis receives cross-paper evidence."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for hit in hits:
        metadata = hit.get("metadata", {})
        paper_key = str(metadata.get("paper_id") or metadata.get("file_name") or "unknown")
        grouped.setdefault(paper_key, []).append(hit)

    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < top_k:
        added = False
        for paper_hits in grouped.values():
            if depth < len(paper_hits):
                selected.append(paper_hits[depth])
                added = True
                if len(selected) >= top_k:
                    break
        if not added:
            break
        depth += 1
    return selected


class PaperRAG:
    def __init__(self, vector_store: ChromaVectorStore, llm_client: OpenAICompatibleClient) -> None:
        self.vector_store = vector_store
        self.llm_client = llm_client

    def answer(
        self,
        question: str,
        paper_id: str | None = None,
        top_k: int | None = None,
        *,
        retrieval_mode: str | None = None,
        expand_parent: bool | None = None,
    ) -> dict[str, Any]:
        """Return an answer and source chunks, using section-aware fallback."""

        hits, analysis = self.retrieve_hits(
            question,
            paper_id=paper_id,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            expand_parent=expand_parent,
        )
        return self.answer_from_hits(question, hits, analysis=analysis)

    def retrieve_hits(
        self,
        question: str,
        paper_id: str | None = None,
        top_k: int | None = None,
        *,
        retrieval_mode: str | None = None,
        expand_parent: bool | None = None,
    ) -> tuple[list[dict[str, Any]], QueryAnalysis]:
        """Retrieve once and return both hits and the analysis used for them."""

        analysis = analyze_question(question)
        retrieval_top_k = top_k if top_k is not None else (
            settings.retrieval_top_k if paper_id else settings.multi_paper_top_k
        )
        multi_paper = paper_id is None and is_multi_paper_question(question)
        query_top_k = retrieval_top_k * 2 if multi_paper else retrieval_top_k
        hits = self.vector_store.query(
            query_text=analysis.rewritten_query,
            embedding_client=self.llm_client,
            top_k=query_top_k,
            paper_id=paper_id,
            section_types=analysis.section_types,
            retrieval_mode=retrieval_mode,
            expand_parent=expand_parent,
            diversify_papers=multi_paper,
        )
        if analysis.section_types and not hits:
            hits = self.vector_store.query(
                query_text=analysis.rewritten_query,
                embedding_client=self.llm_client,
                top_k=query_top_k,
                paper_id=paper_id,
                retrieval_mode=retrieval_mode,
                expand_parent=expand_parent,
                diversify_papers=multi_paper,
            )
        if multi_paper:
            hits = _diversify_hits_by_paper(hits, retrieval_top_k)
        return hits, analysis

    def answer_from_hits(
        self,
        question: str,
        hits: list[dict[str, Any]],
        *,
        analysis: QueryAnalysis | None = None,
    ) -> dict[str, Any]:
        """Generate from already-retrieved hits so evaluation does not retrieve twice."""

        analysis = analysis or analyze_question(question)
        if not hits:
            return {
                "answer": refusal_for_question(question),
                "raw_answer": "",
                "sources": [],
                "query_analysis": analysis.__dict__,
                "refusal_reason": "no_sources",
                "refusal_type": "system_refusal",
                "warning": "未检索到可用来源。",
            }

        messages = build_qa_messages(question, hits)
        qa_chat = getattr(self.llm_client, "chat_qa", self.llm_client.chat)
        raw_answer = qa_chat(messages)
        consistency_retried = False
        if _needs_binary_consistency_retry(question, raw_answer):
            consistency_retried = True
            raw_answer = qa_chat(
                messages
                + [
                    {"role": "assistant", "content": raw_answer},
                    {
                        "role": "user",
                        "content": (
                            "请复核首句的‘是/否’是否与后文不等式和文字解释逻辑一致。"
                            "若 A≥B 与 B≤A 等价，必须使用相同极性。只输出修正后的完整回答。"
                        ),
                    },
                ],
                temperature=0.0,
            )
        citation_retry_attempted = False
        if settings.strict_citation and not _has_citation(raw_answer) and not is_refusal_answer(raw_answer):
            citation_retry_attempted = True
            raw_answer = qa_chat(
                _citation_retry_messages(messages, raw_answer, len(hits)),
                temperature=0.0,
            )
        result = self._finalize_answer(raw_answer, hits, analysis)
        result["binary_consistency_retried"] = consistency_retried
        result["citation_retry_attempted"] = citation_retry_attempted
        if citation_retry_attempted:
            retry_message = (
                "初次回答缺少引用，已自动重新生成带引用答案。"
                if result.get("refusal_reason") is None
                else "自动补充引用后仍未生成合法引用，已触发保守拒答。"
            )
            result["warning"] = " ".join(filter(None, (result.get("warning"), retry_message)))
        result["model_metadata"] = dict(getattr(self.llm_client, "last_chat_metadata", {}) or {})
        return result

    @staticmethod
    def _finalize_answer(
        raw_answer: str,
        hits: list[dict[str, Any]],
        analysis: QueryAnalysis,
    ) -> dict[str, Any]:
        refusal = refusal_for_question(analysis.original_question)
        refusal_type = classify_refusal_answer(raw_answer)
        final_answer = _strip_trailing_refusal(raw_answer) if refusal_type == "mixed_refusal" else raw_answer
        final_answer, invalid_citation_ids = repair_source_citations(final_answer, len(hits))
        citation_repaired = bool(invalid_citation_ids) and _has_citation(final_answer)
        result: dict[str, Any] = {
            "answer": final_answer,
            "raw_answer": raw_answer,
            "sources": hits,
            "query_analysis": analysis.__dict__,
            "refusal_reason": None,
            "refusal_type": refusal_type,
            "citation_valid": (not invalid_citation_ids) if _has_citation(final_answer) else None,
            "invalid_citation_ids": invalid_citation_ids,
            "citation_repaired": citation_repaired,
        }
        if refusal_type == "pure_refusal":
            result["refusal_reason"] = "model_refusal"
            return result
        if settings.strict_citation and invalid_citation_ids and not citation_repaired:
            invalid_labels = ", ".join(f"S{source_id}" for source_id in invalid_citation_ids)
            result.update(
                answer=refusal,
                refusal_reason="invalid_citation",
                refusal_type="system_refusal",
                warning=f"模型引用了不存在的来源 {invalid_labels}；本次仅提供 S1-S{len(hits)}。",
            )
            return result
        if settings.strict_citation and not _has_citation(final_answer):
            result.update(
                answer=refusal,
                refusal_reason="missing_citation",
                refusal_type="system_refusal",
                warning="模型回答未包含可核验引用，已触发保守拒答。",
            )
            return result
        if citation_repaired:
            invalid_labels = ", ".join(f"S{source_id}" for source_id in invalid_citation_ids)
            result["citation_valid"] = True
            result["warning"] = f"已移除不存在的来源 {invalid_labels}，其余有效引用已保留。"
        if refusal_type == "mixed_refusal":
            mixed_warning = "回答同时包含实质内容和拒答语句，已清理尾部拒答并标记为 mixed_refusal。"
            result["warning"] = " ".join(filter(None, (result.get("warning"), mixed_warning)))
        return result

    def answer_from_hits_stream(
        self,
        question: str,
        hits: list[dict[str, Any]],
        *,
        analysis: QueryAnalysis | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield answer deltas followed by one authoritative final result."""

        analysis = analysis or analyze_question(question)
        if not hits:
            yield {
                "event": "done",
                "data": {
                    "answer": refusal_for_question(question),
                    "raw_answer": "",
                    "sources": [],
                    "query_analysis": analysis.__dict__,
                    "refusal_reason": "no_sources",
                    "refusal_type": "system_refusal",
                    "warning": "未检索到可用来源。",
                },
            }
            return

        yield {
            "event": "meta",
            "data": {
                "query_analysis": analysis.__dict__,
                "source_count": len(hits),
            },
        }
        parts: list[str] = []
        qa_stream = getattr(self.llm_client, "chat_stream_qa", self.llm_client.chat_stream)
        for delta in qa_stream(build_qa_messages(question, hits)):
            parts.append(delta)
            yield {"event": "delta", "data": {"text": delta}}

        raw_answer = "".join(parts)
        citation_retry_attempted = False
        if settings.strict_citation and not _has_citation(raw_answer) and not is_refusal_answer(raw_answer):
            citation_retry_attempted = True
            qa_chat = getattr(self.llm_client, "chat_qa", self.llm_client.chat)
            raw_answer = qa_chat(
                _citation_retry_messages(build_qa_messages(question, hits), raw_answer, len(hits)),
                temperature=0.0,
            )
        result = self._finalize_answer(raw_answer, hits, analysis)
        result["citation_retry_attempted"] = citation_retry_attempted
        if citation_retry_attempted:
            retry_message = (
                "初次回答缺少引用，已自动重新生成带引用答案。"
                if result.get("refusal_reason") is None
                else "自动补充引用后仍未生成合法引用，已触发保守拒答。"
            )
            result["warning"] = " ".join(filter(None, (result.get("warning"), retry_message)))
        result["model_metadata"] = dict(getattr(self.llm_client, "last_chat_metadata", {}) or {})
        yield {"event": "done", "data": result}


def format_sources(sources: list[dict[str, Any]]) -> list[str]:
    """Format source chunks for frontend display."""

    formatted: list[str] = []
    for source in sources:
        metadata = source.get("metadata", {})
        page = metadata.get("page", "?")
        parent_pages = metadata.get("parent_pages")
        page_text = f"pages {parent_pages}" if parent_pages else f"page {page}"
        file_name = metadata.get("file_name", "unknown")
        section_path = metadata.get("section_path") or "unknown section"
        section_type = metadata.get("section_type") or "unknown"
        source_id = metadata.get("chunk_id", "")
        retrieval_source = source.get("retrieval_source", "")
        text = source.get("text", "").replace("\n", " ")
        formatted.append(
            f"{file_name} | {page_text} | {section_path} ({section_type}) | "
            f"{retrieval_source} | {source_id} | {text[:300]}"
        )
    return formatted

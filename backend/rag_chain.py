"""RAG question answering: analyze query, retrieve, and answer with citations."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from collections.abc import Iterator
from time import perf_counter
from typing import Any

from backend.config import settings
from backend.embeddings import OpenAICompatibleClient
from backend.prompts import (
    answer_language,
    build_evidence_plan_messages,
    build_qa_messages,
    is_long_form_question,
    is_yes_no_question,
)
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


_VACUOUS_ANSWER_PATTERNS = (
    re.compile(
        r"^\s*(?:是|否)?[，,：:\s]*(?:论文|文中|论文中|来源|片段)(?:中)?(?:有|存在|包含)(?:明确|直接)?(?:的)?(?:依据|证据|信息)"
        r"(?:\s*\[[^\]]+\])?[。.!]?\s*$"
    ),
    re.compile(
        r"^\s*(?:yes|no)?[,：:\s]*(?:the\s+)?(?:paper|sources?|context|passages?).{0,40}"
        r"(?:has|have|contains?|provides?|supports?).{0,30}(?:evidence|support|information)"
        r"(?:\s*\[[^\]]+\])?[.!]?\s*$",
        re.I,
    ),
)


def is_vacuous_answer(answer: str) -> bool:
    """Return whether the model reported evidence existence without answering."""

    normalized = " ".join(answer.strip().split())
    return bool(normalized and any(pattern.fullmatch(normalized) for pattern in _VACUOUS_ANSWER_PATTERNS))


_INTENT_EVIDENCE_TERMS: dict[str, tuple[str, ...]] = {
    "experiment": (
        "experiment", "experimental", "evaluation", "baseline", "benchmark", "parameter",
        "result", "improve", "reduce", "accuracy", "latency", "实验", "评估", "基线", "参数", "结果",
    ),
    "method": ("method", "algorithm", "architecture", "framework", "training", "方法", "算法", "架构", "训练"),
    "background": ("background", "motivation", "challenge", "problem", "背景", "动机", "挑战", "问题"),
    "contribution": ("contribution", "novel", "propose", "innovation", "贡献", "创新", "提出"),
    "related_work": ("related work", "previous", "prior", "literature", "相关工作", "已有工作", "文献"),
}

_INTENT_EVIDENCE_SECTIONS: dict[str, set[str]] = {
    "experiment": {"experiment", "result"},
    "method": {"method"},
    "background": {"abstract", "introduction", "related_work"},
    "contribution": {"abstract", "introduction", "conclusion"},
    "related_work": {"related_work"},
}


def _has_intent_evidence(analysis: QueryAnalysis, hits: list[dict[str, Any]]) -> bool:
    """Use conservative lexical signals to spot an obviously premature refusal."""

    terms = _INTENT_EVIDENCE_TERMS.get(analysis.intent)
    if not terms:
        return False
    expected_sections = _INTENT_EVIDENCE_SECTIONS.get(analysis.intent, set())
    evidence_parts: list[str] = []
    for hit in hits:
        metadata = hit.get("metadata") or {}
        if str(metadata.get("section_type") or "") in expected_sections:
            # Section-aware retrieval already narrowed this source to exactly
            # the part requested by the user. A pure refusal should therefore
            # be challenged even when the prose lacks literal words such as
            # "background" or "motivation".
            return True
        evidence_parts.extend(
            [
                str(hit.get("generation_text") or hit.get("section_document") or hit.get("text") or ""),
                str(metadata.get("section_path") or ""),
                str(metadata.get("section_type") or ""),
            ]
        )
    evidence = " ".join(evidence_parts).lower()
    return sum(1 for term in terms if term in evidence) >= 2


def _answer_quality_retry_reason(
    question: str,
    answer: str,
    analysis: QueryAnalysis,
    hits: list[dict[str, Any]],
) -> str | None:
    if is_vacuous_answer(answer):
        return "vacuous_answer"
    if is_refusal_answer(answer) and _has_intent_evidence(analysis, hits):
        return "premature_refusal"
    return None


def _answer_quality_retry_messages(
    messages: list[dict[str, str]],
    raw_answer: str,
    question: str,
    retry_reason: str,
) -> list[dict[str, str]]:
    problem = (
        "上一条只说明了存在依据，却没有回答问题"
        if retry_reason == "vacuous_answer"
        else "上一条过早拒答，但提供的来源中包含与问题直接相关的候选证据"
    )
    return messages + [
        {"role": "assistant", "content": raw_answer},
        {
            "role": "user",
            "content": (
                f"{problem}。请重新阅读全部来源并直接回答：{question}\n"
                "要求：提取来源中能确认的具体事实、过程、参数、数字和结论；每个关键结论标注来源编号。"
                "不得只回复‘有依据/有证据’，也不要因个别细节缺失而拒绝整个问题；"
                "确实缺少的部分可以省略或单独说明。只输出修正后的完整答案。"
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


def _normalize_open_answer_lead(question: str, answer: str) -> str:
    """Remove a spurious affirmative lead from non-binary content answers."""

    if is_yes_no_question(question):
        return answer
    return re.sub(r"^\s*(?:是[，,：:]|yes[,：:])\s*", "", answer, count=1, flags=re.I).strip()


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


def _deep_retrieval_queries(question: str, analysis: QueryAnalysis) -> list[str]:
    """Expand one complex request into complementary evidence searches."""

    original = analysis.rewritten_query or question
    intent_queries: dict[str, list[str]] = {
        "experiment": [
            "experimental environment implementation dataset parameter settings",
            "baselines comparison methods evaluation metrics",
            "quantitative results performance improvement convergence",
            "ablation sensitivity scalability limitations",
        ],
        "method": [
            "system model problem formulation assumptions variables constraints objective",
            "method architecture framework components workflow",
            "algorithm steps training optimization complexity",
            "design motivation why each module is necessary",
        ],
        "background": [
            "research background application context motivation",
            "existing limitations research gap challenge",
            "problem definition objective contributions",
            "relationship to prior work and practical significance",
        ],
        "contribution": [
            "main contributions novelty proposed framework",
            "problem formulation system model",
            "method design and algorithm workflow",
            "experiments quantitative evidence conclusion limitations",
        ],
        "related_work": [
            "related work research categories prior approaches",
            "limitations of existing methods research gap",
            "difference between proposed method and prior work",
        ],
        "general": [
            "research background motivation problem and contributions",
            "system model assumptions variables constraints optimization objective",
            "method architecture algorithm workflow and design rationale",
            "experimental setup datasets baselines metrics quantitative results",
            "conclusion limitations future work practical implications",
        ],
    }
    # Run the user's resolved query last so retrieval diagnostics continue to
    # expose the actual user intent rather than an internal expansion.
    queries = [*intent_queries.get(analysis.intent, intent_queries["general"]), original]
    return list(dict.fromkeys(" ".join(query.split()) for query in queries if query.strip()))


def _round_robin_unique_hits(
    result_groups: list[list[dict[str, Any]]],
    limit: int,
) -> list[dict[str, Any]]:
    """Interleave retrieval facets while keeping one representative per section."""

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    positions = [0 for _ in result_groups]
    while len(selected) < limit:
        added = False
        for group_index, hits in enumerate(result_groups):
            while positions[group_index] < len(hits):
                hit = hits[positions[group_index]]
                positions[group_index] += 1
                metadata = hit.get("metadata") or {}
                paper = str(metadata.get("paper_id") or metadata.get("file_name") or "")
                section = str(
                    metadata.get("parent_id")
                    or metadata.get("section_path")
                    or metadata.get("chunk_id")
                    or hit.get("text", "")[:120]
                )
                identity = f"{paper}::{section}"
                if identity not in seen:
                    selected.append(hit)
                    seen.add(identity)
                    added = True
                    break
            if len(selected) >= limit:
                break
        if not added:
            break
    return selected


class PaperRAG:
    def __init__(self, vector_store: ChromaVectorStore, llm_client: OpenAICompatibleClient) -> None:
        self.vector_store = vector_store
        self.llm_client = llm_client

    def _prepare_generation(
        self,
        question: str,
        hits: list[dict[str, Any]],
        prompt_question: str,
        *,
        deep_mode: bool | None = None,
    ) -> tuple[list[dict[str, str]], bool, bool]:
        """Create a private evidence outline before complex synthesis."""

        long_form = is_long_form_question(question) if deep_mode is None else bool(deep_mode)
        evidence_plan = ""
        client_config = getattr(self.llm_client, "config", None)
        planning_enabled = bool(
            long_form
            and getattr(client_config, "qa_planning_enabled", False)
        )
        if planning_enabled:
            try:
                plan_messages = build_evidence_plan_messages(prompt_question, hits)
                plan_max_tokens = max(int(getattr(client_config, "qa_plan_max_tokens", 900)), 1)
                try:
                    evidence_plan = self.llm_client.chat(
                        plan_messages,
                        temperature=0.0,
                        max_tokens=plan_max_tokens,
                        model_candidates=self._qa_model_candidates(True),
                        enable_thinking=self._qa_enable_thinking(True),
                    ).strip()
                except TypeError as exc:
                    if "unexpected keyword argument" not in str(exc):
                        raise
                    evidence_plan = self.llm_client.chat(
                        plan_messages,
                        temperature=0.0,
                        max_tokens=plan_max_tokens,
                    ).strip()
            except Exception:
                # Planning improves coverage but must not make QA unavailable
                # when a provider rejects one auxiliary request.
                evidence_plan = ""
        messages = build_qa_messages(
            prompt_question,
            hits,
            evidence_plan=evidence_plan or None,
            long_form=long_form,
        )
        return messages, long_form, bool(evidence_plan)

    def _qa_token_limit(self, long_form: bool) -> int:
        client_config = getattr(self.llm_client, "config", settings)
        setting_name = "qa_long_max_tokens" if long_form else "qa_max_tokens"
        fallback = settings.qa_long_max_tokens if long_form else settings.qa_max_tokens
        return max(int(getattr(client_config, setting_name, fallback)), 1)

    def _qa_model_candidates(self, long_form: bool) -> list[str] | None:
        if not long_form:
            return None
        client_config = getattr(self.llm_client, "config", settings)
        candidates = getattr(client_config, "qa_long_models", None)
        return list(candidates) if candidates else None

    def _qa_enable_thinking(self, long_form: bool) -> bool | None:
        if not long_form:
            return None
        client_config = getattr(self.llm_client, "config", settings)
        return bool(getattr(client_config, "qa_long_enable_thinking", False))

    def _chat_qa(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float = 0.2,
        long_form: bool = False,
    ) -> str:
        qa_chat = getattr(self.llm_client, "chat_qa", self.llm_client.chat)
        try:
            return qa_chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                model_candidates=self._qa_model_candidates(long_form),
                enable_thinking=self._qa_enable_thinking(long_form),
            )
        except TypeError as exc:
            # Test doubles and older custom clients may not yet expose the
            # optional generation keywords.
            if "unexpected keyword argument" not in str(exc):
                raise
            try:
                return qa_chat(messages, temperature=temperature, max_tokens=max_tokens)
            except TypeError as fallback_exc:
                if "unexpected keyword argument" not in str(fallback_exc):
                    raise
                try:
                    return qa_chat(messages, temperature=temperature)
                except TypeError as final_exc:
                    if "unexpected keyword argument" not in str(final_exc):
                        raise
                    return qa_chat(messages)

    def _stream_qa(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float = 0.2,
        long_form: bool = False,
    ) -> Iterator[str]:
        qa_stream = getattr(self.llm_client, "chat_stream_qa", None)
        if qa_stream is None:
            qa_stream = self.llm_client.chat_stream
        try:
            yield from qa_stream(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                model_candidates=self._qa_model_candidates(long_form),
                enable_thinking=self._qa_enable_thinking(long_form),
            )
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            try:
                yield from qa_stream(messages, temperature=temperature, max_tokens=max_tokens)
            except TypeError as fallback_exc:
                if "unexpected keyword argument" not in str(fallback_exc):
                    raise
                try:
                    yield from qa_stream(messages, temperature=temperature)
                except TypeError as final_exc:
                    if "unexpected keyword argument" not in str(final_exc):
                        raise
                    yield from qa_stream(messages)

    def _should_continue(self, long_form: bool) -> bool:
        client_config = getattr(self.llm_client, "config", settings)
        metadata = getattr(self.llm_client, "last_chat_metadata", {}) or {}
        return bool(
            long_form
            and getattr(client_config, "qa_auto_continue", False)
            and str(metadata.get("finish_reason") or "").lower() == "length"
        )

    @staticmethod
    def _continuation_messages(
        messages: list[dict[str, str]],
        partial_answer: str,
    ) -> list[dict[str, str]]:
        return messages + [
            {"role": "assistant", "content": partial_answer},
            {
                "role": "user",
                "content": (
                    "上一段回答因输出长度限制中断。请从中断处继续，补完尚未覆盖的论文、章节、"
                    "比较维度、实验结果和最终综合结论。不要重复已经写过的内容，不要重新编号来源，"
                    "继续只使用原始来源允许的 [S编号]。只输出续写正文。"
                ),
            },
        ]

    def answer(
        self,
        question: str,
        paper_id: str | None = None,
        top_k: int | None = None,
        *,
        retrieval_question: str | None = None,
        retrieval_mode: str | None = None,
        expand_parent: bool | None = None,
        deep_mode: bool | None = None,
    ) -> dict[str, Any]:
        """Return an answer and source chunks, using section-aware fallback."""

        resolved_deep_mode = self.resolve_deep_mode(question, paper_id, deep_mode)
        resolved_question = retrieval_question or question
        hits, retrieval_analysis = self.retrieve_hits(
            resolved_question,
            paper_id=paper_id,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            expand_parent=expand_parent,
            deep_mode=resolved_deep_mode,
        )
        analysis = QueryAnalysis(
            original_question=question,
            rewritten_query=retrieval_analysis.rewritten_query,
            intent=retrieval_analysis.intent,
            section_types=retrieval_analysis.section_types,
        )
        return self.answer_from_hits(
            question,
            hits,
            analysis=analysis,
            resolved_question=resolved_question if retrieval_question else None,
            deep_mode=resolved_deep_mode,
        )

    @staticmethod
    def resolve_deep_mode(
        question: str,
        paper_id: str | None,
        deep_mode: bool | None = None,
    ) -> bool:
        """Resolve the QA profile while always protecting multi-paper synthesis.

        A caller-provided switch controls single-paper QA.  Corpus-wide QA always
        keeps the complete retrieval and evidence-planning path.  Callers that do
        not yet expose a switch retain the historical intent-based behavior.
        """

        if paper_id is None:
            return True
        if deep_mode is not None:
            return bool(deep_mode)
        return is_long_form_question(question)

    def retrieve_hits(
        self,
        question: str,
        paper_id: str | None = None,
        top_k: int | None = None,
        *,
        retrieval_mode: str | None = None,
        expand_parent: bool | None = None,
        deep_mode: bool | None = None,
    ) -> tuple[list[dict[str, Any]], QueryAnalysis]:
        """Retrieve once and return both hits and the analysis used for them."""

        analysis = analyze_question(question)
        deep_retrieval = self.resolve_deep_mode(question, paper_id, deep_mode)
        retrieval_top_k = top_k if top_k is not None else (
            max(settings.qa_detailed_top_k, settings.retrieval_top_k)
            if paper_id and deep_retrieval
            else settings.retrieval_top_k if paper_id else settings.multi_paper_top_k
        )
        multi_paper = paper_id is None
        query_top_k = retrieval_top_k * 2 if multi_paper and not deep_retrieval else retrieval_top_k
        queries = _deep_retrieval_queries(question, analysis) if deep_retrieval else [analysis.rewritten_query]
        per_query_top_k = (
            max(4, min(retrieval_top_k, 8))
            if deep_retrieval
            else query_top_k
        )
        result_groups: list[list[dict[str, Any]]] = []
        timing_totals: dict[str, float] = {}
        retrieval_batch_started = perf_counter()

        def retrieve_one(retrieval_query: str) -> tuple[list[dict[str, Any]], dict[str, float]]:
            query_hits = self.vector_store.query(
                query_text=retrieval_query,
                embedding_client=self.llm_client,
                top_k=per_query_top_k,
                paper_id=paper_id,
                section_types=None if deep_retrieval else analysis.section_types,
                retrieval_mode=retrieval_mode,
                expand_parent=expand_parent,
                diversify_papers=multi_paper,
            )
            query_timings = dict(
                getattr(self.vector_store, "last_query_timings_ms", {}) or {}
            )
            return query_hits, query_timings

        worker_count = min(
            len(queries),
            max(int(getattr(settings, "qa_retrieval_workers", 1)), 1),
        )
        if deep_retrieval and worker_count > 1:
            # executor.map preserves the facet order even when requests finish in
            # a different order, keeping merge and citation behavior deterministic.
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="rag-retrieval",
            ) as executor:
                query_results = list(executor.map(retrieve_one, queries))
        else:
            query_results = [retrieve_one(retrieval_query) for retrieval_query in queries]

        facet_total_ms = 0.0
        for query_hits, query_timings in query_results:
            result_groups.append(query_hits)
            for name, value in query_timings.items():
                if isinstance(value, (int, float)):
                    if name == "total_ms":
                        facet_total_ms += float(value)
                    else:
                        timing_totals[name] = timing_totals.get(name, 0.0) + float(value)
        timing_totals["facet_total_ms"] = facet_total_ms
        timing_totals["total_ms"] = (perf_counter() - retrieval_batch_started) * 1000
        hits = _round_robin_unique_hits(
            result_groups,
            retrieval_top_k * 2 if multi_paper else retrieval_top_k,
        )
        if deep_retrieval and analysis.intent != "related_work":
            non_reference_hits = [
                hit
                for hit in hits
                if str((hit.get("metadata") or {}).get("section_type") or "") != "references"
            ]
            if non_reference_hits:
                hits = non_reference_hits
        if not deep_retrieval and analysis.section_types and not hits:
            hits = self.vector_store.query(
                query_text=analysis.rewritten_query,
                embedding_client=self.llm_client,
                top_k=query_top_k,
                paper_id=paper_id,
                retrieval_mode=retrieval_mode,
                expand_parent=expand_parent,
                diversify_papers=multi_paper,
            )
        if timing_totals:
            self.vector_store.last_query_timings_ms = {
                name: round(value, 2) for name, value in timing_totals.items()
            }
        if multi_paper:
            hits = _diversify_hits_by_paper(hits, retrieval_top_k)
        return hits, analysis

    def answer_from_hits(
        self,
        question: str,
        hits: list[dict[str, Any]],
        *,
        analysis: QueryAnalysis | None = None,
        resolved_question: str | None = None,
        deep_mode: bool | None = None,
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

        prompt_question = self._prompt_question(question, resolved_question)
        messages, long_form, evidence_plan_used = self._prepare_generation(
            question,
            hits,
            prompt_question,
            deep_mode=deep_mode,
        )
        max_tokens = self._qa_token_limit(long_form)
        raw_answer = self._chat_qa(messages, max_tokens=max_tokens, long_form=long_form)
        continuation_attempted = self._should_continue(long_form)
        if continuation_attempted:
            continuation = self._chat_qa(
                self._continuation_messages(messages, raw_answer),
                max_tokens=max_tokens,
                temperature=0.0,
                long_form=long_form,
            )
            if continuation.strip():
                raw_answer = f"{raw_answer.rstrip()}\n\n{continuation.lstrip()}"
        consistency_retried = False
        if _needs_binary_consistency_retry(question, raw_answer):
            consistency_retried = True
            raw_answer = self._chat_qa(
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
                max_tokens=max_tokens,
                temperature=0.0,
                long_form=long_form,
            )
        quality_retry_reason = _answer_quality_retry_reason(question, raw_answer, analysis, hits)
        quality_retry_attempted = quality_retry_reason is not None
        if quality_retry_reason:
            raw_answer = self._chat_qa(
                _answer_quality_retry_messages(
                    messages,
                    raw_answer,
                    question,
                    quality_retry_reason,
                ),
                max_tokens=max_tokens,
                temperature=0.0,
                long_form=long_form,
            )
        citation_retry_attempted = False
        if settings.strict_citation and not _has_citation(raw_answer) and not is_refusal_answer(raw_answer):
            citation_retry_attempted = True
            raw_answer = self._chat_qa(
                _citation_retry_messages(messages, raw_answer, len(hits)),
                max_tokens=max_tokens,
                temperature=0.0,
                long_form=long_form,
            )
        result = self._finalize_answer(raw_answer, hits, analysis)
        result["binary_consistency_retried"] = consistency_retried
        result["answer_quality_retry_attempted"] = quality_retry_attempted
        result["answer_quality_retry_reason"] = quality_retry_reason
        result["citation_retry_attempted"] = citation_retry_attempted
        result["long_form"] = long_form
        result["evidence_plan_used"] = evidence_plan_used
        result["continuation_attempted"] = continuation_attempted
        result["max_output_tokens"] = max_tokens
        if quality_retry_attempted:
            quality_message = (
                "初次回答未提供实质内容，已自动重新生成。"
                if quality_retry_reason == "vacuous_answer"
                else "初次回答可能过早拒答，已基于检索证据自动复核。"
            )
            result["warning"] = " ".join(filter(None, (result.get("warning"), quality_message)))
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
        final_answer = _normalize_open_answer_lead(analysis.original_question, final_answer)
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
        resolved_question: str | None = None,
        deep_mode: bool | None = None,
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
            "event": "stage",
            "data": {
                "stage": "reranking",
                "label": "重排中",
                "message": "正在筛选证据并组织回答结构…",
            },
        }
        prompt_question = self._prompt_question(question, resolved_question)
        messages, long_form, evidence_plan_used = self._prepare_generation(
            question,
            hits,
            prompt_question,
            deep_mode=deep_mode,
        )
        max_tokens = self._qa_token_limit(long_form)
        yield {
            "event": "stage",
            "data": {
                "stage": "generating",
                "label": "生成中",
                "message": "正在根据论文证据生成回答…",
            },
        }
        yield {
            "event": "meta",
            "data": {
                "query_analysis": analysis.__dict__,
                "source_count": len(hits),
                "long_form": long_form,
                "evidence_plan_used": evidence_plan_used,
                "max_output_tokens": max_tokens,
            },
        }
        parts: list[str] = []
        for delta in self._stream_qa(messages, max_tokens=max_tokens, long_form=long_form):
            parts.append(delta)
            yield {"event": "delta", "data": {"text": delta}}

        raw_answer = "".join(parts)
        continuation_attempted = self._should_continue(long_form)
        if continuation_attempted:
            for delta in self._stream_qa(
                self._continuation_messages(messages, raw_answer),
                max_tokens=max_tokens,
                temperature=0.0,
                long_form=long_form,
            ):
                parts.append(delta)
                yield {"event": "delta", "data": {"text": delta}}
            raw_answer = "".join(parts)
        quality_retry_reason = _answer_quality_retry_reason(question, raw_answer, analysis, hits)
        quality_retry_attempted = quality_retry_reason is not None
        if quality_retry_reason:
            raw_answer = self._chat_qa(
                _answer_quality_retry_messages(
                    messages,
                    raw_answer,
                    question,
                    quality_retry_reason,
                ),
                max_tokens=max_tokens,
                temperature=0.0,
                long_form=long_form,
            )
        citation_retry_attempted = False
        if settings.strict_citation and not _has_citation(raw_answer) and not is_refusal_answer(raw_answer):
            citation_retry_attempted = True
            raw_answer = self._chat_qa(
                _citation_retry_messages(messages, raw_answer, len(hits)),
                max_tokens=max_tokens,
                temperature=0.0,
                long_form=long_form,
            )
        result = self._finalize_answer(raw_answer, hits, analysis)
        result["answer_quality_retry_attempted"] = quality_retry_attempted
        result["answer_quality_retry_reason"] = quality_retry_reason
        result["citation_retry_attempted"] = citation_retry_attempted
        result["long_form"] = long_form
        result["evidence_plan_used"] = evidence_plan_used
        result["continuation_attempted"] = continuation_attempted
        result["max_output_tokens"] = max_tokens
        if quality_retry_attempted:
            quality_message = (
                "初次回答未提供实质内容，已自动重新生成。"
                if quality_retry_reason == "vacuous_answer"
                else "初次回答可能过早拒答，已基于检索证据自动复核。"
            )
            result["warning"] = " ".join(filter(None, (result.get("warning"), quality_message)))
        if citation_retry_attempted:
            retry_message = (
                "初次回答缺少引用，已自动重新生成带引用答案。"
                if result.get("refusal_reason") is None
                else "自动补充引用后仍未生成合法引用，已触发保守拒答。"
            )
            result["warning"] = " ".join(filter(None, (result.get("warning"), retry_message)))
        result["model_metadata"] = dict(getattr(self.llm_client, "last_chat_metadata", {}) or {})
        yield {"event": "done", "data": result}

    @staticmethod
    def _prompt_question(question: str, resolved_question: str | None) -> str:
        if not resolved_question or resolved_question.strip() == question.strip():
            return question
        return (
            f"当前用户追问：{question}\n"
            f"结合上一轮上下文解析后的完整问题：{resolved_question}\n"
            "请回答当前追问；解析后的完整问题仅用于补足省略的主题和指代。"
        )


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

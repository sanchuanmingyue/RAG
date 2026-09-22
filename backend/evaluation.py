"""Offline evaluation helpers for PaperReader-RAG.

The module keeps benchmark data, evaluation collections, and reports outside the
interactive Streamlit flow.  It currently adapts Vectara Open RAG Benchmark's
``queries.json``/``qrels.json``/``answers.json`` layout, while the metric code
is intentionally reusable for a hand-written project-specific golden set.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
import random
import re
import statistics
import time
from typing import Any, Callable, Iterable
import unicodedata

from backend.config import ROOT_DIR
from backend.embeddings import OpenAICompatibleClient
from backend.rag_chain import PaperRAG, classify_refusal_answer, extract_source_citation_ids, is_refusal_answer
from backend.text_splitter import TextChunk, classify_section_title, split_text
from backend.vector_store import ChromaVectorStore


EVAL_RESULTS_DIR = ROOT_DIR / "storage" / "evals"
_CITATION_PATTERN = re.compile(r"\[(?=[^\]]*\bS\s*\d)[^\]]+\]", flags=re.I)
_MARKDOWN_HEADING_PATTERN = re.compile(r"#{1,6}\s+(.+?)(?=(?:\s+#{1,6}\s+)|\r?\n|$)")


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    question: str
    gold_doc_id: str
    gold_section_id: str
    reference_answer: str = ""
    source_type: str = ""
    query_type: str = ""


@dataclass(frozen=True)
class EvaluationConfig:
    top_k: int = 5
    retrieval_mode: str = "hybrid"
    expand_parent: bool = False
    run_generation: bool = False
    corpus_document_count: int = 0
    embedding_model: str = ""
    collection_name: str = ""
    chunk_strategy: str = "section_paragraph"
    chunk_size: int = 0
    chunk_overlap: int = 0
    two_stage_retrieval: bool = True
    document_candidate_k: int = 50
    document_top_k: int = 5
    section_candidate_k: int = 50
    section_support_chunks: int = 3
    reference_section_penalty: float = 0.35
    section_index: bool = True
    keyword_query_cache_size: int = 64
    cross_encoder_reranker: bool = True
    cross_encoder_model: str = ""
    cross_encoder_candidate_k: int = 20
    cross_encoder_weight: float = 0.65
    section_reranker_provider: str = ""
    section_reranker_model: str = ""
    section_reranker_candidate_k: int = 20
    section_reranker_weight: float = 0.65
    answer_correctness_method: str = "staged_hybrid_v1"
    answer_correctness_threshold: float = 0.70
    answer_correctness_borderline_low: float = 0.60
    answer_correctness_borderline_high: float = 0.75
    answer_judge_enabled: bool = True
    answer_judge_models: tuple[str, ...] = ()
    claim_citation_eval_enabled: bool = True
    claim_citation_similarity_threshold: float = 0.55


@dataclass
class EvaluationRecord:
    case_id: str
    question: str
    gold_doc_id: str
    gold_section_id: str
    source_type: str
    query_type: str
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    doc_rank: int | None = None
    section_rank: int | None = None
    retrieval_latency_ms: float = 0.0
    retrieval_stage_latency_ms: dict[str, float] = field(default_factory=dict)
    generation_latency_ms: float | None = None
    answer: str = ""
    raw_answer: str = ""
    reference_answer: str = ""
    citation_valid: bool | None = None
    citation_gold_doc_hit: bool | None = None
    citation_gold_section_hit: bool | None = None
    llm_model_used: str | None = None
    llm_attempted_models: list[str] = field(default_factory=list)
    llm_fallback_count: int = 0
    refused: bool = False
    refusal_reason: str | None = None
    refusal_type: str | None = None
    invalid_citation_ids: list[int] = field(default_factory=list)
    citation_repaired: bool = False
    binary_consistency_retried: bool = False
    generation_warning: str | None = None
    answer_similarity: float | None = None
    answer_correct: bool | None = None
    answer_correctness_stage: str | None = None
    answer_correctness_reason: str | None = None
    answer_correctness_error: str | None = None
    judge_model_used: str | None = None
    judge_reason: str | None = None
    judge_raw_output: str | None = None
    claim_count: int = 0
    cited_claim_count: int = 0
    supported_claim_count: int = 0
    citation_link_count: int = 0
    supported_citation_link_count: int = 0
    claim_citation_precision: float | None = None
    claim_citation_recall: float | None = None
    unsupported_claim_rate: float | None = None
    claim_citation_details: list[dict[str, Any]] = field(default_factory=list)
    claim_citation_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_dataset_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Open RAG Benchmark 下载目录不存在：{root.resolve()}。"
            "请先下载数据集，或将 --dataset-root 指向实际下载目录。"
        )
    candidates = (
        root,
        root / "official" / "pdf" / "arxiv",
        root / "pdf" / "arxiv",
    )
    for candidate in candidates:
        if all((candidate / file_name).is_file() for file_name in ("queries.json", "qrels.json", "answers.json")):
            return candidate
    raise FileNotFoundError(
        f"在 {root.resolve()} 下未找到 Open RAG Benchmark 文件。dataset_root 应包含 "
        "queries.json、qrels.json、answers.json，或指向包含 official/pdf/arxiv、pdf/arxiv 的下载目录。"
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_query(raw_query: Any) -> tuple[str, str, str]:
    if isinstance(raw_query, str):
        return raw_query, "", ""
    if not isinstance(raw_query, dict):
        return "", "", ""
    return (
        str(raw_query.get("query") or raw_query.get("question") or "").strip(),
        str(raw_query.get("source") or "").strip(),
        str(raw_query.get("type") or "").strip(),
    )


def _coerce_qrel(raw_qrel: Any) -> tuple[str, str]:
    if isinstance(raw_qrel, dict):
        doc_id = raw_qrel.get("doc_id")
        section_id = raw_qrel.get("section_id")
        return (
            "" if doc_id is None else str(doc_id).strip(),
            "" if section_id is None else str(section_id).strip(),
        )
    if isinstance(raw_qrel, list) and raw_qrel and isinstance(raw_qrel[0], dict):
        return _coerce_qrel(raw_qrel[0])
    return "", ""


def load_open_rag_bench_cases(
    dataset_root: str | Path,
    *,
    max_cases: int | None = None,
    seed: int = 42,
    text_only: bool = True,
) -> list[BenchmarkCase]:
    """Load a deterministic sample of Open RAG Benchmark cases.

    Text-only filtering is enabled by default because the main project RAG
    pipeline processes PDF text.  Table/image cases should be evaluated through
    the separate RAG-Anything workflow.
    """

    root = _resolve_dataset_root(dataset_root)
    queries = _read_json(root / "queries.json")
    qrels = _read_json(root / "qrels.json")
    answers = _read_json(root / "answers.json")
    if not isinstance(queries, dict) or not isinstance(qrels, dict) or not isinstance(answers, dict):
        raise ValueError("Open RAG Benchmark 的 queries/qrels/answers 文件应为 JSON 对象。")

    cases: list[BenchmarkCase] = []
    for case_id, raw_query in queries.items():
        question, source_type, query_type = _coerce_query(raw_query)
        doc_id, section_id = _coerce_qrel(qrels.get(case_id))
        if not question or not doc_id:
            continue
        if text_only and source_type and source_type != "text":
            continue
        cases.append(
            BenchmarkCase(
                case_id=str(case_id),
                question=question,
                gold_doc_id=doc_id,
                gold_section_id=section_id,
                reference_answer=str(answers.get(case_id) or ""),
                source_type=source_type,
                query_type=query_type,
            )
        )

    cases.sort(key=lambda case: case.case_id)
    if max_cases is not None and max_cases < len(cases):
        if max_cases <= 0:
            raise ValueError("max_cases 必须大于 0。")
        cases = sorted(random.Random(seed).sample(cases, max_cases), key=lambda case: case.case_id)
    return cases


def _safe_chunk_prefix(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", value).strip("_")
    return cleaned or "document"


def _section_text(section: Any, include_tables: bool) -> str:
    if not isinstance(section, dict):
        return ""
    parts = [str(section.get("text") or "").strip()]
    if include_tables:
        tables = section.get("tables")
        if isinstance(tables, dict):
            parts.extend(str(table).strip() for table in tables.values())
    return "\n\n".join(part for part in parts if part)


def _markdown_section_metadata(text: str, section_id: str) -> tuple[str, str, str]:
    """Extract the heading chain embedded at the start of benchmark Markdown."""

    prefix = text.lstrip()
    headings: list[str] = []
    if prefix.startswith("#"):
        first_line = prefix.splitlines()[0]
        headings = [
            re.sub(r"\s+", " ", match.group(1)).strip(" #")
            for match in _MARKDOWN_HEADING_PATTERN.finditer(first_line)
            if match.group(1).strip(" #")
        ]
    fallback = f"Benchmark section {section_id}"
    title = headings[-1] if headings else fallback
    section_path = " > ".join(headings) if headings else fallback
    section_type = "benchmark_section"
    for heading in reversed(headings):
        classified = classify_section_title(heading)
        if classified:
            section_type = classified
            break
    return title, section_path, section_type


def build_open_rag_bench_chunks(
    dataset_root: str | Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
    include_tables: bool = False,
    document_ids: set[str] | None = None,
) -> list[TextChunk]:
    """Convert benchmark corpus sections to this project's TextChunk format.

    This is the *retrieval-only* benchmark mode: it uses the benchmark's own
    parsed corpus so that qrels section IDs can be measured exactly.  A later
    raw-PDF mode can be added to assess parsing quality separately.
    """

    root = _resolve_dataset_root(dataset_root)
    corpus_dir = root / "corpus"
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"未找到 benchmark corpus：{corpus_dir}")

    chunks: list[TextChunk] = []
    for corpus_file in sorted(corpus_dir.glob("*.json")):
        # Benchmark corpus filenames are document IDs. Skip unselected files
        # before JSON parsing so a 200-document evaluation does not read the
        # complete corpus merely to discard most of it.
        if document_ids is not None and corpus_file.stem not in document_ids:
            continue
        paper = _read_json(corpus_file)
        if not isinstance(paper, dict):
            continue
        doc_id = str(paper.get("id") or corpus_file.stem).strip()
        if document_ids is not None and doc_id not in document_ids:
            continue
        title = str(paper.get("title") or doc_id).strip()
        sections = paper.get("sections")
        if not doc_id or not isinstance(sections, list):
            continue

        prefix = _safe_chunk_prefix(doc_id)
        for section_index, section in enumerate(sections):
            text = _section_text(section, include_tables=include_tables)
            if not text:
                continue
            raw_section_id = section.get("section_id", section_index) if isinstance(section, dict) else section_index
            section_id = str(section_index if raw_section_id is None else raw_section_id)
            section_title, section_path, section_type = _markdown_section_metadata(text, section_id)
            parent_id = f"orb_{prefix}_section_{_safe_chunk_prefix(section_id)}"
            for child_index, child_text in enumerate(
                split_text(text, chunk_size, chunk_overlap),
                start=1,
            ):
                chunks.append(
                    TextChunk(
                        chunk_id=f"{parent_id}_chunk_{child_index}",
                        paper_id=doc_id,
                        file_name=f"{title}.json",
                        page=section_index + 1,
                        text=child_text,
                        section_title=section_title,
                        section_type=section_type,
                        section_path=section_path,
                        parent_id=parent_id,
                        parent_index=section_index + 1,
                        child_index=child_index,
                        benchmark_doc_id=doc_id,
                        benchmark_section_id=section_id,
                    )
                )
    if not chunks:
        raise ValueError("benchmark corpus 中没有可索引的文本章节。")
    return chunks


def select_open_rag_bench_document_ids(
    dataset_root: str | Path,
    cases: list[BenchmarkCase],
    *,
    max_documents: int = 200,
    seed: int = 42,
) -> set[str]:
    """Build a reproducible mini-corpus with every gold document plus negatives.

    ``max_documents=0`` means the full corpus.  A smaller corpus is appropriate
    for rapid configuration comparisons; use the full corpus for a final score.
    """

    root = _resolve_dataset_root(dataset_root)
    corpus_dir = root / "corpus"
    # Open RAG Benchmark names each corpus JSON file with its document ID.
    # Using the stem avoids parsing the entire (large) corpus before a small
    # benchmark run; selected documents are parsed later during indexing.
    all_document_ids = {corpus_file.stem for corpus_file in corpus_dir.glob("*.json") if corpus_file.stem}

    gold_document_ids = {case.gold_doc_id for case in cases if case.gold_doc_id in all_document_ids}
    if not all_document_ids:
        raise ValueError("benchmark corpus 中没有可用文档。")
    if max_documents < 0:
        raise ValueError("max_documents 不能小于 0。")
    if max_documents == 0 or max_documents >= len(all_document_ids):
        return all_document_ids

    desired_count = max(max_documents, len(gold_document_ids))
    remaining = sorted(all_document_ids - gold_document_ids)
    sampled = random.Random(seed).sample(remaining, min(desired_count - len(gold_document_ids), len(remaining)))
    return gold_document_ids | set(sampled)


def index_chunks_in_batches(
    vector_store: ChromaVectorStore,
    chunks: list[TextChunk],
    embedding_client: OpenAICompatibleClient,
    *,
    batch_size: int = 200,
    resume: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    """Index in bounded batches and resume safely after an interrupted run."""

    indexed = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        existing_ids: set[str] = set()
        collection = getattr(vector_store, "collection", None)
        if resume and collection is not None:
            existing = collection.get(ids=[chunk.chunk_id for chunk in batch], include=[])
            existing_ids = {str(value) for value in existing.get("ids", [])}
        missing = [chunk for chunk in batch if chunk.chunk_id not in existing_ids]
        if missing:
            vector_store.index_chunks(missing, embedding_client)
        indexed += len(batch)
        if progress_callback is not None:
            progress_callback(indexed, len(chunks))
    return indexed


def _rank_for(hit_list: Iterable[dict[str, Any]], gold_doc_id: str, gold_section_id: str) -> tuple[int | None, int | None]:
    doc_rank: int | None = None
    section_rank: int | None = None
    for rank, hit in enumerate(hit_list, start=1):
        metadata = hit.get("metadata", {})
        doc_id = str(metadata.get("benchmark_doc_id") or metadata.get("paper_id") or "")
        section_id = str(metadata.get("benchmark_section_id") or "")
        if doc_rank is None and doc_id == gold_doc_id:
            doc_rank = rank
        if section_rank is None and doc_id == gold_doc_id and section_id == gold_section_id:
            section_rank = rank
    return doc_rank, section_rank


def _serialize_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        item = dict(hit)
        item["rank"] = rank
        item["text"] = str(item.get("text") or "")[:500]
        serialized.append(item)
    return serialized


def _citation_validity(answer: str, source_count: int) -> bool | None:
    cited_ids = extract_source_citation_ids(answer)
    if not cited_ids:
        return None
    return bool(source_count) and all(1 <= source_id <= source_count for source_id in cited_ids)


def _citation_gold_alignment(
    answer: str,
    sources: list[dict[str, Any]],
    gold_doc_id: str,
    gold_section_id: str,
) -> tuple[bool | None, bool | None]:
    """Check whether at least one cited source points to the benchmark gold target."""

    cited_ids = extract_source_citation_ids(answer)
    if not cited_ids or not sources or any(source_id < 1 or source_id > len(sources) for source_id in cited_ids):
        return None, None
    cited_sources = [sources[source_id - 1] for source_id in cited_ids]
    cited_metadata = [source.get("metadata", {}) for source in cited_sources]
    doc_hit = any(str(metadata.get("benchmark_doc_id") or metadata.get("paper_id") or "") == gold_doc_id for metadata in cited_metadata)
    section_hit = any(
        str(metadata.get("benchmark_doc_id") or metadata.get("paper_id") or "") == gold_doc_id
        and str(metadata.get("benchmark_section_id") or "") == gold_section_id
        for metadata in cited_metadata
    )
    return doc_hit, section_hit


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embedding vectors must have the same non-zero dimension")
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / denominator


def _answer_for_similarity(answer: str) -> str:
    """Remove citation syntax that should not influence semantic correctness."""

    return _CITATION_PATTERN.sub(" ", answer).strip()


def _extract_polarity(text: str) -> bool | None:
    """Extract an explicit leading yes/no answer across English and Chinese."""

    normalized = _answer_for_similarity(text).strip().lstrip("*#>- ").lower()
    negative_patterns = (
        r"^(?:no|false)\b",
        r"^(?:答案)?(?:是|为)?(?:否定|错误)(?:的)?(?=[，。,.!！：:\s]|$)",
        r"^(?:否|不是|并非|并不是|不需要|无需|不会|不能|无法|不可以|没有|无)(?=[，。,.!！：:\s]|$)",
        r"^(?:(?:该|此|这种|上述)\s*)?(?:方法|模型|系统|论文|研究|算法|框架|技术|数据集|过程|机制)"
        r"(?:并)?(?:不(?!仅)|未|没有|无法|不能|不会|无需|并非|不是)",
    )
    positive_patterns = (
        r"^(?:yes|true)\b",
        r"^(?:答案)?(?:是|为)?(?:肯定|正确)(?:的)?(?=[，。,.!！：:\s]|$)",
        r"^(?:是的|是|可以|会|需要|包含|有|能够|正确)(?=[，。,.!！：:\s]|$)",
    )
    if any(re.search(pattern, normalized, flags=re.I) for pattern in negative_patterns):
        return False
    if any(re.search(pattern, normalized, flags=re.I) for pattern in positive_patterns):
        return True

    # Chinese answers often start with evidence attribution ("根据论文片段")
    # and place the actual negative conclusion in the middle of the sentence.
    # Prefer the final explicit conclusion clause, then fall back to the whole
    # answer. Exclude common additive constructs such as “不仅/不但”.
    conclusion_parts = re.split(r"(?:因此|所以|综上(?:所述)?|由此可见|结论(?:是|为)?)[，,:：\s]*", normalized)
    search_spaces = [conclusion_parts[-1], normalized] if len(conclusion_parts) > 1 else [normalized]
    middle_negative = re.compile(
        r"(?:并不(?:总是)?|通常不|往往不|不(?!仅|但)|未(?:能|曾|被|使用|采用|包含|提供|发现|显示)?|"
        r"没有|无法|不能|不会|无需|不需要|并非|不是)"
    )
    if any(middle_negative.search(search_space) for search_space in search_spaces):
        return False
    return None


def _normalize_exact(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", _answer_for_similarity(text)).lower()
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    return re.sub(r"[^0-9a-z\u4e00-\u9fff%+\-=<>\\^{}_.]+", "", normalized)


def _normalize_number(value: str) -> str:
    normalized = value.replace(",", "").lower().lstrip("+")
    if normalized.startswith("."):
        normalized = f"0{normalized}"
    return normalized


def _normalize_proper_name(value: str) -> str:
    without_article = re.sub(r"^(?:the|a|an)\s+", "", value.strip(), flags=re.I)
    return _normalize_exact(without_article)


_NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.])[+-]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?%?"
    r"(?![A-Za-z0-9_]|\.\d)",
    re.I,
)


def _extract_numbers(text: str) -> list[str]:
    """Extract numbers next to Chinese text without accepting partial numbers."""

    normalized = unicodedata.normalize("NFKC", text)
    return [_normalize_number(value) for value in _NUMBER_PATTERN.findall(normalized)]


def _structured_targets(question: str, reference: str) -> tuple[str | None, list[str]]:
    """Extract high-confidence formula, numeric, or proper-name answer targets."""

    formula_matches = re.findall(
        r"\$\$?(.+?)\$\$?|\\\((.+?)\\\)|\\\[(.+?)\\\]",
        reference,
        flags=re.S,
    )
    formulas = [next(value for value in match if value) for match in formula_matches]
    if formulas and re.search(r"formula|equation|expression|bound|公式|方程|表达式|上界|下界", question, re.I):
        return "formula", [_normalize_exact(value) for value in formulas]

    numbers = _extract_numbers(reference)
    if numbers and re.search(
        r"how many|how much|what (?:is )?(?:the )?(?:value|number|rate|ratio|percentage|bound)|"
        r"多少|数值|比例|百分比|上界|下界",
        question,
        re.I,
    ):
        return "numeric", list(dict.fromkeys(numbers))

    question_requests_name = re.search(
        r"what (?:is|are|does|type of)?\s*(?:the )?(?:method|model|algorithm|framework|dataset|problem|metric|"
        r"operation|equation|distribution|architecture|name|type)|which (?:method|model|algorithm|framework|dataset|operation)|"
        r"什么(?:方法|模型|算法|框架|数据集|问题|指标|操作|方程|分布|架构|名称|类型)",
        question,
        re.I,
    )
    if re.search(r"\bwhat\s+(?:is\s+)?(?:the\s+)?problem\b|什么问题", question, re.I):
        question_requests_name = None
    if question_requests_name and len(reference.split()) <= 25:
        quoted = [value for pair in re.findall(r"`([^`]+)`|[\"“]([^\"”]+)[\"”]", reference) for value in pair if value]
        acronyms = re.findall(r"\b[A-Z][A-Z0-9-]{1,}\b", reference)
        camel_case = re.findall(r"\b[A-Z][A-Za-z0-9-]*[A-Z][A-Za-z0-9-]*\b", reference)
        title_sequences = re.findall(r"\b(?:[A-Z][a-z]+(?:[- ][A-Z][a-z]+)+)\b", reference)
        targets = list(dict.fromkeys(quoted + acronyms + camel_case + title_sequences))
        if targets:
            return "proper_noun", [_normalize_proper_name(value) for value in targets]
    return None, []


def _structured_match(kind: str, targets: list[str], answer: str) -> bool:
    normalized_answer = _normalize_exact(answer)
    if kind == "numeric":
        answer_numbers = set(_extract_numbers(answer))
        return all(target in answer_numbers for target in targets)
    return all(target and target in normalized_answer for target in targets)


def _judge_messages(record: EvaluationRecord) -> list[dict[str, str]]:
    payload = {
        "question": record.question,
        "reference_answer": record.reference_answer,
        "generated_answer": _answer_for_similarity(record.answer),
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a strict answer-equivalence judge. Determine whether the generated answer is semantically "
                "consistent with the reference answer for the question. Accept different languages, wording, and "
                "additional correct explanation. Reject contradictions, wrong polarity/numbers/formulas, or missing "
                "essential facts. Output only JSON: {\"correct\": true|false, \"reason\": \"brief reason\"}."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _parse_judge_output(output: str) -> tuple[bool, str]:
    candidate = output.strip().lstrip("\ufeff")
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.I | re.S)
    if fenced:
        candidate = fenced.group(1).strip()

    candidates = [candidate]
    decoder = json.JSONDecoder()
    for index, character in enumerate(candidate):
        if character != "{":
            continue
        try:
            _value, consumed = decoder.raw_decode(candidate[index:])
            candidates.append(candidate[index : index + consumed])
        except json.JSONDecodeError:
            continue
    object_match = re.search(r"\{.*\}", candidate, flags=re.S)
    if object_match:
        candidates.append(object_match.group(0))

    for raw in candidates:
        variants = [raw, re.sub(r",\s*([}\]])", r"\1", raw)]
        for variant in variants:
            data: Any
            try:
                data = json.loads(variant)
            except (json.JSONDecodeError, TypeError):
                try:
                    data = ast.literal_eval(variant)
                except (ValueError, SyntaxError):
                    continue
            if not isinstance(data, dict):
                continue
            correct = data.get("correct", data.get("is_correct", data.get("verdict")))
            if isinstance(correct, str):
                lowered = correct.strip().lower()
                if lowered in {"true", "yes", "correct", "1"}:
                    correct = True
                elif lowered in {"false", "no", "incorrect", "0"}:
                    correct = False
            elif isinstance(correct, int) and correct in {0, 1}:
                correct = bool(correct)
            if isinstance(correct, bool):
                return correct, str(data.get("reason") or data.get("explanation") or "").strip()

    boolean_match = re.search(
        r"(?:correct|is_correct|verdict)\s*[\"']?\s*[:=]\s*[\"']?(true|false|yes|no|1|0)\b",
        candidate,
        flags=re.I,
    )
    if boolean_match:
        correct = boolean_match.group(1).lower() in {"true", "yes", "1"}
        reason_match = re.search(r"reason\s*[\"']?\s*[:=]\s*[\"']([^\"'\r\n}]+)", candidate, re.I)
        return correct, reason_match.group(1).strip() if reason_match else ""
    raise ValueError("judge output does not contain a recognizable boolean 'correct' field")


def score_answer_correctness(
    records: list[EvaluationRecord],
    embedding_client: OpenAICompatibleClient,
    *,
    threshold: float,
    borderline_low: float,
    borderline_high: float,
    judge_enabled: bool,
    judge_models: list[str] | None = None,
) -> None:
    """Apply polarity, structured, semantic, then judge-based correctness scoring."""

    eligible = [
        record
        for record in records
        if record.generation_latency_ms is not None and record.reference_answer.strip()
    ]
    for record in eligible:
        if record.refused:
            record.answer_similarity = 0.0
            record.answer_correct = False
            record.answer_correctness_stage = "refusal"
            record.answer_correctness_reason = record.refusal_reason or "refused"

    answered = [record for record in eligible if not record.refused and record.answer.strip()]
    semantic_records: list[EvaluationRecord] = []
    for record in answered:
        reference_polarity = _extract_polarity(record.reference_answer)
        if reference_polarity is not None:
            answer_polarity = _extract_polarity(record.answer)
            record.answer_correct = answer_polarity is not None and answer_polarity == reference_polarity
            record.answer_correctness_stage = "yes_no"
            record.answer_correctness_reason = (
                f"reference={reference_polarity}, answer={answer_polarity}"
            )
            continue

        structured_kind, targets = _structured_targets(record.question, record.reference_answer)
        if structured_kind and targets:
            structured_match = _structured_match(structured_kind, targets, record.answer)
            if structured_kind == "proper_noun" and not structured_match:
                semantic_records.append(record)
                continue
            record.answer_correct = structured_match
            record.answer_correctness_stage = "structured_exact"
            record.answer_correctness_reason = f"{structured_kind}: {targets}"
            continue
        semantic_records.append(record)

    if not semantic_records:
        return
    texts = [_answer_for_similarity(record.answer) for record in semantic_records]
    texts.extend(record.reference_answer.strip() for record in semantic_records)
    try:
        embeddings = embedding_client.embed_texts(texts)
        split_at = len(semantic_records)
        for record, answer_embedding, reference_embedding in zip(
            semantic_records,
            embeddings[:split_at],
            embeddings[split_at:],
        ):
            similarity = max(-1.0, min(1.0, _cosine_similarity(answer_embedding, reference_embedding)))
            record.answer_similarity = round(similarity, 6)
            if similarity < borderline_low or similarity > borderline_high or not judge_enabled:
                record.answer_correct = similarity >= threshold
                record.answer_correctness_stage = "semantic_similarity"
                record.answer_correctness_reason = f"cosine={similarity:.6f}"
                continue
            try:
                judge_output = embedding_client.chat(
                    _judge_messages(record),
                    temperature=0.0,
                    model_candidates=judge_models or None,
                )
                record.judge_raw_output = judge_output
                record.answer_correct, record.judge_reason = _parse_judge_output(judge_output)
                record.judge_model_used = embedding_client.last_chat_metadata.get("model_used")
                record.answer_correctness_stage = "llm_judge"
                record.answer_correctness_reason = f"borderline cosine={similarity:.6f}"
            except Exception as exc:
                record.answer_correct = similarity >= threshold
                record.answer_correctness_stage = "semantic_similarity_fallback"
                record.answer_correctness_reason = f"judge failed; cosine={similarity:.6f}"
                record.answer_correctness_error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        for record in semantic_records:
            record.answer_correctness_error = error


def _split_answer_claims(answer: str) -> list[str]:
    """Split prose/bullets into citation-bearing claim units.

    A citation at the end of a paragraph/list item applies to every claim in
    that same block.  It never crosses a newline, which keeps neighbouring
    bullets and paragraphs independent.
    """

    claims: list[str] = []
    for raw_block in answer.splitlines():
        block = raw_block.strip().lstrip("-*#> ")
        if not block:
            continue

        trailing_ids: list[int] = []
        citation_matches = list(_CITATION_PATTERN.finditer(block))
        if citation_matches:
            trailing_start = citation_matches[-1].start()
            suffix = block[citation_matches[-1].end():]
            if not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", suffix):
                for match in reversed(citation_matches[:-1]):
                    between = block[match.end():trailing_start]
                    if re.search(r"[A-Za-z0-9\u4e00-\u9fff]", between):
                        break
                    trailing_start = match.start()
                trailing_ids = list(dict.fromkeys(extract_source_citation_ids(block[trailing_start:])))

        block_claims: list[str] = []
        raw_parts = re.split(r"(?<=[。！？!?])|(?<=\.)\s+", block)
        for raw_part in raw_parts:
            part = raw_part.strip()
            if not part:
                continue
            without_citations = _CITATION_PATTERN.sub("", part).strip()
            content_size = len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", without_citations))

            # Headings, lead-ins, and list labels such as "主要结论：" do not
            # make a factual claim on their own.
            if without_citations.rstrip().endswith((":", "：")):
                continue
            if content_size < 4 and extract_source_citation_ids(part) and block_claims:
                block_claims[-1] = f"{block_claims[-1]} {part}".strip()
                continue
            if content_size < 8:
                continue
            if trailing_ids and not extract_source_citation_ids(part):
                labels = ", ".join(f"S{source_id}" for source_id in trailing_ids)
                part = f"{part} [{labels}]"
            block_claims.append(part)
        claims.extend(block_claims)
    return claims


def score_claim_citations(
    records: list[EvaluationRecord],
    sources_by_case: dict[str, list[dict[str, Any]]],
    embedding_client: OpenAICompatibleClient,
    *,
    similarity_threshold: float,
) -> None:
    """Estimate claim-level citation support with cross-language embeddings.

    This is a semantic-support proxy rather than a logical entailment model. It
    reports citation-link precision, supported-claim recall, and unsupported
    claim rate so citation presence alone cannot mask unsupported assertions.
    """

    work: dict[str, tuple[list[str], list[dict[str, Any]]]] = {}
    unique_texts: dict[str, None] = {}
    for record in records:
        if record.generation_latency_ms is None or record.refused or not record.answer.strip():
            continue
        claims = _split_answer_claims(record.answer)
        sources = sources_by_case.get(record.case_id, [])
        work[record.case_id] = (claims, sources)
        record.claim_count = len(claims)
        for claim in claims:
            claim_text = _answer_for_similarity(claim)
            if claim_text:
                unique_texts[claim_text] = None
            for source_id in extract_source_citation_ids(claim):
                if 1 <= source_id <= len(sources):
                    source = sources[source_id - 1]
                    source_text = str(source.get("section_document") or source.get("text") or "")[:3000]
                    if source_text:
                        unique_texts[source_text] = None

    if not unique_texts:
        return
    texts = list(unique_texts)
    try:
        vectors = embedding_client.embed_texts(texts)
        vector_by_text = dict(zip(texts, vectors))
        for record in records:
            item = work.get(record.case_id)
            if item is None:
                continue
            claims, sources = item
            supported_claims = 0
            details: list[dict[str, Any]] = []
            for claim in claims:
                cited_ids = list(
                    dict.fromkeys(
                        source_id
                        for source_id in extract_source_citation_ids(claim)
                        if 1 <= source_id <= len(sources)
                    )
                )
                if cited_ids:
                    record.cited_claim_count += 1
                claim_text = _answer_for_similarity(claim)
                claim_vector = vector_by_text.get(claim_text)
                scores: list[float] = []
                for source_id in cited_ids:
                    source = sources[source_id - 1]
                    source_text = str(source.get("section_document") or source.get("text") or "")[:3000]
                    source_vector = vector_by_text.get(source_text)
                    if claim_vector is None or source_vector is None:
                        continue
                    score = max(-1.0, min(1.0, _cosine_similarity(claim_vector, source_vector)))
                    scores.append(score)
                    record.citation_link_count += 1
                    if score >= similarity_threshold:
                        record.supported_citation_link_count += 1
                supported = bool(scores) and max(scores) >= similarity_threshold
                if supported:
                    supported_claims += 1
                details.append(
                    {
                        "claim": claim[:300],
                        "citation_ids": cited_ids,
                        "max_similarity": round(max(scores), 6) if scores else None,
                        "supported": supported,
                    }
                )
            record.claim_citation_details = details
            record.supported_claim_count = supported_claims
            if record.citation_link_count:
                record.claim_citation_precision = round(
                    record.supported_citation_link_count / record.citation_link_count,
                    4,
                )
            if record.claim_count:
                record.claim_citation_recall = round(supported_claims / record.claim_count, 4)
                record.unsupported_claim_rate = round(1.0 - supported_claims / record.claim_count, 4)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        for record in records:
            if record.case_id in work:
                record.claim_citation_error = error


def run_evaluation(
    cases: list[BenchmarkCase],
    vector_store: ChromaVectorStore,
    embedding_client: OpenAICompatibleClient,
    config: EvaluationConfig,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[EvaluationRecord]:
    """Execute retrieval and, optionally, the existing PaperRAG answer path."""

    records: list[EvaluationRecord] = []
    sources_by_case: dict[str, list[dict[str, Any]]] = {}
    rag = PaperRAG(vector_store, embedding_client) if config.run_generation else None
    for case in cases:
        started = time.perf_counter()
        hits = vector_store.query(
            query_text=case.question,
            embedding_client=embedding_client,
            top_k=config.top_k,
            retrieval_mode=config.retrieval_mode,
            expand_parent=config.expand_parent,
        )
        retrieval_latency_ms = (time.perf_counter() - started) * 1000
        doc_rank, section_rank = _rank_for(hits, case.gold_doc_id, case.gold_section_id)
        record = EvaluationRecord(
            case_id=case.case_id,
            question=case.question,
            gold_doc_id=case.gold_doc_id,
            gold_section_id=case.gold_section_id,
            source_type=case.source_type,
            query_type=case.query_type,
            retrieved=_serialize_hits(hits),
            doc_rank=doc_rank,
            section_rank=section_rank,
            retrieval_latency_ms=retrieval_latency_ms,
            retrieval_stage_latency_ms=dict(vector_store.last_query_timings_ms),
            reference_answer=case.reference_answer,
        )
        if rag is not None:
            sources_by_case[case.case_id] = hits
            started = time.perf_counter()
            result = rag.answer_from_hits(case.question, hits)
            record.generation_latency_ms = (time.perf_counter() - started) * 1000
            record.answer = str(result["answer"])
            record.raw_answer = str(result.get("raw_answer") or "")
            citation_answer = record.answer
            explicit_citation_valid = result.get("citation_valid")
            record.citation_valid = (
                explicit_citation_valid
                if explicit_citation_valid is not None
                else _citation_validity(citation_answer, len(result["sources"]))
            )
            record.citation_gold_doc_hit, record.citation_gold_section_hit = _citation_gold_alignment(
                citation_answer,
                result["sources"],
                case.gold_doc_id,
                case.gold_section_id,
            )
            chat_metadata = embedding_client.last_chat_metadata
            record.llm_model_used = chat_metadata.get("model_used")
            record.llm_attempted_models = list(chat_metadata.get("attempted_models") or [])
            record.llm_fallback_count = int(chat_metadata.get("fallback_count") or 0)
            record.refusal_reason = result.get("refusal_reason")
            record.refusal_type = result.get("refusal_type") or classify_refusal_answer(record.answer)
            record.invalid_citation_ids = list(result.get("invalid_citation_ids") or [])
            record.citation_repaired = bool(result.get("citation_repaired"))
            record.binary_consistency_retried = bool(result.get("binary_consistency_retried"))
            record.generation_warning = result.get("warning")
            record.refused = bool(record.refusal_reason) or is_refusal_answer(record.answer)
            if record.refused and not record.refusal_reason:
                record.refusal_reason = "model_refusal"
        records.append(record)
        if progress_callback is not None:
            progress_callback(len(records), len(cases))
    if config.run_generation:
        score_answer_correctness(
            records,
            embedding_client,
            threshold=config.answer_correctness_threshold,
            borderline_low=config.answer_correctness_borderline_low,
            borderline_high=config.answer_correctness_borderline_high,
            judge_enabled=config.answer_judge_enabled,
            judge_models=list(config.answer_judge_models),
        )
        if config.claim_citation_eval_enabled:
            score_claim_citations(
                records,
                sources_by_case,
                embedding_client,
                similarity_threshold=config.claim_citation_similarity_threshold,
            )
    return records


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * percentile)))
    return round(ordered[position], 2)


def summarize_records(
    records: list[EvaluationRecord],
    top_k: int,
    *,
    answer_correctness_threshold: float | None = None,
) -> dict[str, Any]:
    """Calculate retrieval, citation, refusal, and latency summaries."""

    total = len(records)
    if not total:
        return {"case_count": 0, "top_k": top_k}

    doc_hits = [record.doc_rank for record in records if record.doc_rank is not None and record.doc_rank <= top_k]
    section_hits = [record.section_rank for record in records if record.section_rank is not None and record.section_rank <= top_k]
    doc_rr = [1 / record.doc_rank if record.doc_rank else 0.0 for record in records]
    section_rr = [1 / record.section_rank if record.section_rank else 0.0 for record in records]
    generation_records = [record for record in records if record.generation_latency_ms is not None]
    answer_records = [record for record in generation_records if not record.refused]
    citations = [record.citation_valid for record in answer_records]
    valid_citation_records = [record for record in answer_records if record.citation_valid is True]
    model_usage: dict[str, int] = {}
    refusal_reason_counts: dict[str, int] = {}
    refusal_type_counts: dict[str, int] = {}
    for record in generation_records:
        if record.llm_model_used:
            model_usage[record.llm_model_used] = model_usage.get(record.llm_model_used, 0) + 1
        if record.refused:
            reason = record.refusal_reason or "unknown"
            refusal_reason_counts[reason] = refusal_reason_counts.get(reason, 0) + 1
        if record.refusal_type:
            refusal_type_counts[record.refusal_type] = refusal_type_counts.get(record.refusal_type, 0) + 1
    correctness_records = [record for record in generation_records if record.answer_correct is not None]
    similarity_records = [record for record in correctness_records if record.answer_similarity is not None]
    answered_similarity_records = [record for record in similarity_records if not record.refused]
    correctness_stage_counts: dict[str, int] = {}
    judge_model_usage: dict[str, int] = {}
    for record in correctness_records:
        stage = record.answer_correctness_stage or "unknown"
        correctness_stage_counts[stage] = correctness_stage_counts.get(stage, 0) + 1
        if record.judge_model_used:
            judge_model_usage[record.judge_model_used] = judge_model_usage.get(record.judge_model_used, 0) + 1
    generation_latencies = [record.generation_latency_ms for record in generation_records if record.generation_latency_ms is not None]
    claim_records = [record for record in answer_records if record.claim_count > 0 and not record.claim_citation_error]
    total_claims = sum(record.claim_count for record in claim_records)
    total_supported_claims = sum(record.supported_claim_count for record in claim_records)
    total_citation_links = sum(record.citation_link_count for record in claim_records)
    total_supported_links = sum(record.supported_citation_link_count for record in claim_records)
    stage_names = sorted({name for record in records for name in record.retrieval_stage_latency_ms})
    stage_latency = {
        stage_name: {
            "p50": _percentile(
                [record.retrieval_stage_latency_ms[stage_name] for record in records if stage_name in record.retrieval_stage_latency_ms],
                0.5,
            ),
            "p95": _percentile(
                [record.retrieval_stage_latency_ms[stage_name] for record in records if stage_name in record.retrieval_stage_latency_ms],
                0.95,
            ),
        }
        for stage_name in stage_names
    }

    return {
        "case_count": total,
        "top_k": top_k,
        "doc_recall_at_k": round(len(doc_hits) / total, 4),
        "section_recall_at_k": round(len(section_hits) / total, 4),
        "doc_mrr": round(statistics.fmean(doc_rr), 4),
        "section_mrr": round(statistics.fmean(section_rr), 4),
        "citation_presence_rate": (
            round(sum(citation is not None for citation in citations) / len(answer_records), 4) if answer_records else None
        ),
        "citation_valid_rate": (
            round(sum(citation is True for citation in citations) / len(answer_records), 4) if answer_records else None
        ),
        "citation_gold_doc_hit_rate": (
            round(sum(record.citation_gold_doc_hit is True for record in valid_citation_records) / len(valid_citation_records), 4)
            if valid_citation_records else None
        ),
        "citation_gold_section_hit_rate": (
            round(sum(record.citation_gold_section_hit is True for record in valid_citation_records) / len(valid_citation_records), 4)
            if valid_citation_records else None
        ),
        "llm_model_usage": model_usage,
        "llm_fallback_total": sum(record.llm_fallback_count for record in generation_records),
        "answer_correctness_method": "staged_hybrid_v1" if correctness_records else None,
        "answer_correctness_threshold": answer_correctness_threshold if correctness_records else None,
        "answer_correctness_scored_count": len(correctness_records),
        "answer_correctness_stage_counts": correctness_stage_counts,
        "judge_model_usage": judge_model_usage,
        "answer_correct_rate": (
            round(sum(record.answer_correct is True for record in correctness_records) / len(correctness_records), 4)
            if correctness_records else None
        ),
        "answer_semantic_similarity_mean": (
            round(statistics.fmean(record.answer_similarity for record in similarity_records), 4)
            if similarity_records else None
        ),
        "answered_semantic_similarity_mean": (
            round(statistics.fmean(record.answer_similarity for record in answered_similarity_records), 4)
            if answered_similarity_records else None
        ),
        "answer_correctness_error_count": sum(bool(record.answer_correctness_error) for record in generation_records),
        "refusal_rate": (
            round(sum(record.refused for record in generation_records) / len(generation_records), 4)
            if generation_records
            else None
        ),
        "refusal_reason_counts": refusal_reason_counts,
        "refusal_type_counts": refusal_type_counts,
        "citation_repair_count": sum(record.citation_repaired for record in generation_records),
        "binary_consistency_retry_count": sum(record.binary_consistency_retried for record in generation_records),
        "claim_citation_method": "embedding_cosine_proxy_v1" if claim_records else None,
        "claim_citation_precision": (
            round(total_supported_links / total_citation_links, 4) if total_citation_links else None
        ),
        "claim_citation_recall": (
            round(total_supported_claims / total_claims, 4) if total_claims else None
        ),
        "unsupported_claim_rate": (
            round(1.0 - total_supported_claims / total_claims, 4) if total_claims else None
        ),
        "claim_citation_scored_count": len(claim_records),
        "claim_citation_error_count": sum(bool(record.claim_citation_error) for record in generation_records),
        "retrieval_latency_ms": {
            "p50": _percentile([record.retrieval_latency_ms for record in records], 0.5),
            "p95": _percentile([record.retrieval_latency_ms for record in records], 0.95),
        },
        "retrieval_stage_latency_ms": stage_latency,
        "generation_latency_ms": {
            "p50": _percentile(generation_latencies, 0.5),
            "p95": _percentile(generation_latencies, 0.95),
        },
    }


def write_evaluation_report(
    records: list[EvaluationRecord],
    config: EvaluationConfig,
    *,
    run_name: str,
    results_dir: str | Path = EVAL_RESULTS_DIR,
) -> Path:
    """Write a self-contained JSON report that can be compared across runs."""

    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^0-9A-Za-z_-]+", "_", run_name).strip("_") or "evaluation"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{safe_name}_{timestamp}.json"
    payload = {
        "run_name": safe_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": asdict(config),
        "summary": summarize_records(
            records,
            config.top_k,
            answer_correctness_threshold=config.answer_correctness_threshold,
        ),
        "records": [record.to_dict() for record in records],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

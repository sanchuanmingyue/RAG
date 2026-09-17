"""Small, presentation-oriented evaluations for paper-reading RAG systems."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
import random
import re
import statistics
import time
from typing import Any, Iterable

from backend.config import ROOT_DIR, settings
from backend.embeddings import OpenAICompatibleClient
from backend.evaluation import EvaluationRecord, score_answer_correctness, score_claim_citations
from backend.prompts import build_context
from backend.rag_chain import PaperRAG, extract_source_citation_ids, is_refusal_answer
from backend.text_splitter import TextChunk, classify_section_title, split_text
from backend.vector_store import ChromaVectorStore


RESULTS_DIR = ROOT_DIR / "storage" / "evals"
_WORD_PATTERN = re.compile(r"[a-z0-9]+", re.I)


@dataclass(frozen=True)
class LiteratureCase:
    case_id: str
    question: str
    reference_answer: str
    paper_ids: tuple[str, ...]
    gold_evidence: tuple[str, ...] = ()
    unanswerable: bool = False
    suite: str = ""


@dataclass
class LiteratureResult:
    case_id: str
    suite: str
    question: str
    reference_answer: str
    unanswerable: bool
    answer: str = ""
    raw_answer: str = ""
    refused: bool = False
    refusal_reason: str | None = None
    citation_valid: bool | None = None
    cited_source_count: int = 0
    evidence_recall_at_5: float | None = None
    answer_correct: bool | None = None
    answer_correctness_stage: str | None = None
    answer_similarity: float | None = None
    faithfulness: float | None = None
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    sources: list[dict[str, Any]] = field(default_factory=list)


def resolve_qasper_file(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if not candidate.exists():
        raise FileNotFoundError(f"QASPER 路径不存在：{candidate.resolve()}")
    preferred = (
        "qasper-dev-v0.3.json",
        "qasper-test-v0.3.json",
        "qasper-dev-v0.1.json",
    )
    for file_name in preferred:
        matches = list(candidate.rglob(file_name))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"{candidate.resolve()} 中只有数据加载脚本或 dummy 数据，没有正式 QASPER split。"
        "请放入 qasper-dev-v0.3.json（推荐）或直接把 --dataset 指向该文件。"
    )


def _qasper_answer_text(answer: dict[str, Any]) -> str:
    spans = [str(value).strip() for value in answer.get("extractive_spans") or [] if str(value).strip()]
    if spans:
        return ", ".join(spans)
    free_form = str(answer.get("free_form_answer") or "").strip()
    if free_form:
        return free_form
    yes_no = answer.get("yes_no")
    if yes_no is True:
        return "Yes"
    if yes_no is False:
        return "No"
    return ""


def _sample_cases(cases: list[LiteratureCase], max_cases: int, seed: int) -> list[LiteratureCase]:
    ordered = sorted(cases, key=lambda case: case.case_id)
    if max_cases <= 0 or max_cases >= len(ordered):
        return ordered
    return sorted(random.Random(seed).sample(ordered, max_cases), key=lambda case: case.case_id)


def load_qasper_cases(
    dataset: str | Path,
    *,
    max_cases: int = 30,
    seed: int = 42,
    unanswerable_only: bool = False,
) -> tuple[list[LiteratureCase], dict[str, dict[str, Any]]]:
    path = resolve_qasper_file(dataset)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("QASPER JSON 顶层应为 paper_id -> paper 的对象。")

    cases: list[LiteratureCase] = []
    for paper_id, paper in data.items():
        if not isinstance(paper, dict):
            continue
        for qa in paper.get("qas") or []:
            annotations = [item.get("answer", {}) for item in qa.get("answers") or [] if isinstance(item, dict)]
            if not annotations:
                continue
            unanimously_unanswerable = all(bool(answer.get("unanswerable")) for answer in annotations)
            if unanswerable_only != unanimously_unanswerable:
                continue
            answerable = [answer for answer in annotations if not answer.get("unanswerable")]
            reference = "Unanswerable" if unanimously_unanswerable else _qasper_answer_text(answerable[0])
            if not reference:
                continue
            evidence: list[str] = []
            for answer in answerable:
                for value in answer.get("evidence") or []:
                    text = str(value).strip()
                    if text and "FLOAT SELECTED" not in text and text not in evidence:
                        evidence.append(text)
            cases.append(
                LiteratureCase(
                    case_id=str(qa.get("question_id") or f"{paper_id}-{len(cases)}"),
                    question=str(qa.get("question") or "").strip(),
                    reference_answer=reference,
                    paper_ids=(str(paper_id),),
                    gold_evidence=tuple(evidence),
                    unanswerable=unanimously_unanswerable,
                    suite="qasper_unanswerable" if unanswerable_only else "qasper",
                )
            )
    selected = _sample_cases([case for case in cases if case.question], max_cases, seed)
    selected_papers = {paper_id for case in selected for paper_id in case.paper_ids}
    return selected, {str(key): value for key, value in data.items() if str(key) in selected_papers}


def build_qasper_chunks(papers: dict[str, dict[str, Any]]) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for paper_id, paper in papers.items():
        title = str(paper.get("title") or paper_id).strip()
        sections: list[tuple[str, list[str]]] = []
        abstract = str(paper.get("abstract") or "").strip()
        if abstract:
            sections.append(("Abstract", [abstract]))
        for section in paper.get("full_text") or []:
            if not isinstance(section, dict):
                continue
            sections.append(
                (
                    str(section.get("section_name") or "Untitled section").strip(),
                    [str(value).strip() for value in section.get("paragraphs") or [] if str(value).strip()],
                )
            )
        for section_index, (section_title, paragraphs) in enumerate(sections):
            parent_id = f"qasper_{paper_id}_section_{section_index}"
            child_index = 0
            for paragraph in paragraphs:
                for child_text in split_text(paragraph, settings.chunk_size, settings.chunk_overlap):
                    child_index += 1
                    chunks.append(
                        TextChunk(
                            chunk_id=f"{parent_id}_chunk_{child_index}",
                            paper_id=paper_id,
                            file_name=title,
                            page=section_index + 1,
                            text=child_text,
                            section_title=section_title,
                            section_path=section_title,
                            section_type=classify_section_title(section_title) or "paper_section",
                            parent_id=parent_id,
                            parent_index=section_index + 1,
                            child_index=child_index,
                            benchmark_doc_id=paper_id,
                            benchmark_section_id=str(section_index),
                        )
                    )
    return chunks


def load_scholarqa_multi_cases(
    dataset: str | Path,
    *,
    max_cases: int = 10,
    seed: int = 42,
) -> tuple[list[LiteratureCase], list[TextChunk]]:
    path = Path(dataset)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("ScholarQA-Multi human_answers.json 顶层应为数组。")
    rows = [row for row in data if isinstance(row, dict) and row.get("input") and row.get("ctxs")]
    chosen_rows = rows if max_cases <= 0 or max_cases >= len(rows) else random.Random(seed).sample(rows, max_cases)
    chosen_rows.sort(key=lambda row: str(row.get("id") or ""))

    cases: list[LiteratureCase] = []
    chunks: list[TextChunk] = []
    for row_index, row in enumerate(chosen_rows):
        case_id = str(row.get("id") or f"scholarqa-{row_index}")
        allowed_papers: list[str] = []
        for context_index, context in enumerate(row.get("ctxs") or []):
            if not isinstance(context, dict):
                continue
            paper_id = f"{case_id}::paper::{context_index}"
            allowed_papers.append(paper_id)
            title = str(context.get("title") or f"Source {context_index + 1}").strip()
            text = str(context.get("text") or "").strip()
            parent_id = f"scholarqa_{case_id}_source_{context_index}"
            for child_index, child_text in enumerate(
                split_text(text, settings.chunk_size, settings.chunk_overlap), start=1
            ):
                chunks.append(
                    TextChunk(
                        chunk_id=f"{parent_id}_chunk_{child_index}",
                        paper_id=paper_id,
                        file_name=title,
                        page=1,
                        text=child_text,
                        section_title=title,
                        section_path=title,
                        section_type="source",
                        parent_id=parent_id,
                        parent_index=context_index + 1,
                        child_index=child_index,
                        benchmark_doc_id=paper_id,
                        benchmark_section_id="0",
                    )
                )
        cases.append(
            LiteratureCase(
                case_id=case_id,
                question=str(row["input"]).strip(),
                reference_answer=str(row.get("output") or "").strip(),
                paper_ids=tuple(allowed_papers),
                suite="scholarqa_multi",
            )
        )
    return cases, chunks


def _normalized_words(text: str) -> list[str]:
    return _WORD_PATTERN.findall(text.lower())


def evidence_recall(gold_evidence: Iterable[str], hits: list[dict[str, Any]]) -> float | None:
    evidence = [value for value in gold_evidence if value.strip()]
    if not evidence:
        return None
    context_words = _normalized_words(build_context(hits))
    context_set = set(context_words)
    matched = 0
    for value in evidence:
        gold_words = _normalized_words(value)
        if gold_words and len(context_set.intersection(gold_words)) / len(set(gold_words)) >= 0.6:
            matched += 1
    return round(matched / len(evidence), 4)


def _percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * ratio + 0.999999) - 1))
    return round(ordered[index], 2)


def run_literature_evaluation(
    cases: list[LiteratureCase],
    vector_store: ChromaVectorStore,
    client: OpenAICompatibleClient,
    *,
    top_k: int = 5,
    retrieval_mode: str = "hybrid",
    correctness_threshold: float | None = None,
    correctness_borderline_low: float | None = None,
    correctness_borderline_high: float | None = None,
) -> tuple[list[LiteratureResult], dict[str, Any]]:
    rag = PaperRAG(vector_store, client)
    results: list[LiteratureResult] = []
    eval_records: list[EvaluationRecord] = []
    sources_by_case: dict[str, list[dict[str, Any]]] = {}
    for index, case in enumerate(cases, start=1):
        started = time.perf_counter()
        hits = vector_store.query(
            case.question,
            client,
            top_k=top_k,
            paper_id=case.paper_ids[0] if len(case.paper_ids) == 1 else None,
            paper_ids=list(case.paper_ids) if len(case.paper_ids) > 1 else None,
            retrieval_mode=retrieval_mode,
            expand_parent=False,
        )
        retrieval_latency = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        generated = rag.answer_from_hits(case.question, hits)
        generation_latency = (time.perf_counter() - started) * 1000
        answer = str(generated.get("answer") or "")
        refused = bool(generated.get("refusal_reason")) or is_refusal_answer(answer)
        cited_ids = [value for value in dict.fromkeys(extract_source_citation_ids(answer)) if 1 <= value <= len(hits)]
        result = LiteratureResult(
            case_id=case.case_id,
            suite=case.suite,
            question=case.question,
            reference_answer=case.reference_answer,
            unanswerable=case.unanswerable,
            answer=answer,
            raw_answer=str(generated.get("raw_answer") or ""),
            refused=refused,
            refusal_reason=generated.get("refusal_reason"),
            citation_valid=generated.get("citation_valid"),
            cited_source_count=len(cited_ids),
            evidence_recall_at_5=evidence_recall(case.gold_evidence, hits[:top_k]),
            retrieval_latency_ms=round(retrieval_latency, 2),
            generation_latency_ms=round(generation_latency, 2),
            sources=[
                {
                    "rank": rank,
                    "text": str(hit.get("generation_text") or hit.get("text") or "")[:500],
                    "metadata": hit.get("metadata", {}),
                }
                for rank, hit in enumerate(hits, start=1)
            ],
        )
        results.append(result)
        record = EvaluationRecord(
            case_id=case.case_id,
            question=case.question,
            gold_doc_id=case.paper_ids[0] if case.paper_ids else "",
            gold_section_id="",
            source_type="text",
            query_type="literature_qa",
            generation_latency_ms=generation_latency,
            answer=answer,
            raw_answer=result.raw_answer,
            reference_answer=case.reference_answer,
            refused=refused,
            refusal_reason=result.refusal_reason,
        )
        eval_records.append(record)
        sources_by_case[case.case_id] = hits
        print(f"  评测进度：{index}/{len(cases)}", flush=True)

    answerable_records = [record for record, case in zip(eval_records, cases) if not case.unanswerable]
    threshold = settings.answer_correctness_threshold if correctness_threshold is None else correctness_threshold
    borderline_low = (
        settings.answer_correctness_borderline_low
        if correctness_borderline_low is None
        else correctness_borderline_low
    )
    borderline_high = (
        settings.answer_correctness_borderline_high
        if correctness_borderline_high is None
        else correctness_borderline_high
    )
    if answerable_records:
        score_answer_correctness(
            answerable_records,
            client,
            threshold=threshold,
            borderline_low=borderline_low,
            borderline_high=borderline_high,
            judge_enabled=settings.answer_judge_enabled,
            judge_models=tuple(settings.answer_judge_models),
        )
        if settings.claim_citation_eval_enabled:
            score_claim_citations(
                answerable_records,
                sources_by_case,
                client,
                similarity_threshold=settings.claim_citation_similarity_threshold,
            )
    by_id = {record.case_id: record for record in eval_records}
    for result in results:
        record = by_id[result.case_id]
        result.answer_correct = record.answer_correct
        result.answer_correctness_stage = record.answer_correctness_stage
        result.answer_similarity = record.answer_similarity
        result.faithfulness = record.claim_citation_precision

    answerable = [result for result in results if not result.unanswerable]
    unanswerable = [result for result in results if result.unanswerable]
    evidence_scores = [result.evidence_recall_at_5 for result in answerable if result.evidence_recall_at_5 is not None]
    faithfulness_scores = [result.faithfulness for result in answerable if result.faithfulness is not None]
    end_to_end = [result.retrieval_latency_ms + result.generation_latency_ms for result in results]
    summary = {
        "suite": cases[0].suite if cases else None,
        "case_count": len(results),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "answer_correctness_threshold": threshold if answerable else None,
        "answer_correctness_borderline_low": borderline_low if answerable else None,
        "answer_correctness_borderline_high": borderline_high if answerable else None,
        "evidence_recall_at_5": round(statistics.fmean(evidence_scores), 4) if evidence_scores else None,
        "answer_correctness": (
            round(sum(result.answer_correct is True for result in answerable) / len(answerable), 4)
            if answerable else None
        ),
        "faithfulness": round(statistics.fmean(faithfulness_scores), 4) if faithfulness_scores else None,
        "citation_valid_rate": (
            round(sum(result.citation_valid is True for result in answerable) / len(answerable), 4)
            if answerable else None
        ),
        "multi_source_answer_rate": (
            round(sum(result.cited_source_count >= 2 for result in answerable) / len(answerable), 4)
            if answerable else None
        ),
        "correct_refusal_rate": (
            round(sum(result.refused for result in unanswerable) / len(unanswerable), 4)
            if unanswerable else None
        ),
        "false_answer_rate": (
            round(sum(not result.refused for result in unanswerable) / len(unanswerable), 4)
            if unanswerable else None
        ),
        "end_to_end_latency_ms": {"p50": _percentile(end_to_end, 0.5), "p95": _percentile(end_to_end, 0.95)},
    }
    return results, summary


def write_literature_report(
    results: list[LiteratureResult], summary: dict[str, Any], *, run_name: str
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^0-9A-Za-z_-]+", "_", run_name).strip("_") or "literature_eval"
    path = RESULTS_DIR / f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "run_name": safe_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "records": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

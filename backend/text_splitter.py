"""Text splitting helpers for PDF pages and chunk metadata."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TextChunk:
    chunk_id: str
    paper_id: str
    file_name: str
    page: int
    text: str
    section_title: str = ""
    subsection_title: str = ""
    section_type: str = ""
    section_path: str = ""
    parent_id: str = ""
    parent_index: int = 0
    child_index: int = 0
    # 基准测试语料可额外记录标准文档与章节，普通 PDF 入库时保持为空。
    benchmark_doc_id: str = ""
    benchmark_section_id: str = ""

    def to_metadata(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "file_name": self.file_name,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "section_title": self.section_title,
            "subsection_title": self.subsection_title,
            "section_type": self.section_type,
            "section_path": self.section_path,
            "parent_id": self.parent_id,
            "parent_index": self.parent_index,
            "child_index": self.child_index,
            "benchmark_doc_id": self.benchmark_doc_id,
            "benchmark_section_id": self.benchmark_section_id,
        }

    def to_embedding_text(self, include_section_context: bool = True) -> str:
        """Return the text sent to the embedding model without polluting sources.

        The original chunk remains the Chroma ``document`` returned to the user
        and the LLM.  At index time, paper and section labels act as lightweight
        semantic context so questions such as "实验设置" can prefer the relevant
        section even when the body wording is paraphrased.
        """

        if not include_section_context:
            return self.text

        labels: list[str] = []
        if self.file_name:
            labels.append(f"Document: {self.file_name}")
        if self.section_path:
            labels.append(f"Section: {self.section_path}")
        elif self.section_type:
            labels.append(f"Section type: {self.section_type}")
        if not labels:
            return self.text
        return "\n".join(labels) + "\n\n" + self.text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    """Split text with the existing sliding-window strategy."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    cleaned = text.strip()
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append(cleaned[start:end].strip())
        if end == len(cleaned):
            break
        start = end - chunk_overlap

    return [chunk for chunk in chunks if chunk]


SECTION_PATTERNS: list[tuple[str, str, tuple[str, ...]]] = [
    ("abstract", "Abstract", ("abstract",)),
    ("introduction", "Introduction", ("introduction",)),
    (
        "related_work",
        "Related Work",
        (
            "related work",
            "related works",
            "prior work",
            "previous work",
            "literature review",
            "background",
        ),
    ),
    (
        "method",
        "Method",
        (
            "method",
            "methods",
            "methodology",
            "approach",
            "proposed method",
            "proposed approach",
            "model",
            "framework",
            "algorithm",
            "system overview",
            "materials and methods",
        ),
    ),
    (
        "experiment",
        "Experiments",
        (
            "experiment",
            "experiments",
            "experimental setup",
            "experimental settings",
            "evaluation",
            "empirical evaluation",
            "implementation details",
            "dataset",
            "datasets",
            "data sets",
            "benchmarks",
        ),
    ),
    (
        "result",
        "Results",
        (
            "result",
            "results",
            "results and discussion",
            "discussion",
            "analysis",
            "ablation",
            "ablation study",
            "quantitative results",
            "qualitative results",
        ),
    ),
    (
        "conclusion",
        "Conclusion",
        (
            "conclusion",
            "conclusions",
            "conclusion and future work",
            "future work",
        ),
    ),
    ("references", "References", ("references", "bibliography")),
]


_PREFIX_HEADINGS = sorted(
    (
        (phrase, section_type, title)
        for section_type, title, phrases in SECTION_PATTERNS
        for phrase in phrases
    ),
    key=lambda item: len(item[0]),
    reverse=True,
)


def _empty_section() -> dict[str, str]:
    return {
        "section_title": "",
        "subsection_title": "",
        "section_type": "",
        "section_path": "",
    }


def _normalize_heading_text(text: str) -> str:
    text = text.strip()
    text = re.sub(
        r"^\s*(?:section\s+)?(?:[ivxlcdm]+|\d+(?:\.\d+)*|[a-z])[\).\s:-]+",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .:-").lower()


def _heading_level(text: str) -> int:
    match = re.match(r"^\s*(?:section\s+)?(\d+(?:\.\d+)*)[\).\s:-]+", text, flags=re.I)
    if match:
        return match.group(1).count(".") + 1
    if re.match(r"^\s*[A-Z][\).\s:-]+", text):
        return 2
    return 1


def _match_section_heading(line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped or len(stripped) > 180:
        return None

    normalized = _normalize_heading_text(stripped)
    for section_type, title, phrases in SECTION_PATTERNS:
        if normalized in phrases:
            return {
                "title": stripped.strip(" .:-"),
                "section_type": section_type,
                "canonical_title": title,
                "level": _heading_level(stripped),
                "remainder": "",
            }

    # PDF text extraction sometimes returns "Abstract This paper ..." as one
    # line. Keep the remainder in the new section block after recognizing the
    # heading prefix.
    unnumbered = re.sub(
        r"^\s*(?:section\s+)?(?:[ivxlcdm]+|\d+(?:\.\d+)*|[a-z])[\).\s:-]+",
        "",
        stripped,
        flags=re.I,
    )
    lowered = unnumbered.lower()
    for phrase, section_type, title in _PREFIX_HEADINGS:
        prefix = f"{phrase} "
        if lowered.startswith(prefix) and len(unnumbered) > len(prefix) + 20:
            remainder = unnumbered[len(prefix) :].strip()
            return {
                "title": title,
                "section_type": section_type,
                "canonical_title": title,
                "level": _heading_level(stripped),
                "remainder": remainder,
            }

    return None


def classify_section_title(title: str, default: str = "") -> str:
    """Map a real Markdown/PDF heading to the project's canonical section type."""

    normalized = _normalize_heading_text(title)
    for section_type, _canonical_title, phrases in SECTION_PATTERNS:
        if normalized in phrases:
            return section_type
        if any(normalized.startswith(f"{phrase} ") for phrase in phrases):
            return section_type
    return default


def _with_section_path(section: dict[str, str]) -> dict[str, str]:
    section_title = section.get("section_title", "")
    subsection_title = section.get("subsection_title", "")
    if section_title and subsection_title:
        section_path = f"{section_title} > {subsection_title}"
    else:
        section_path = section_title or subsection_title
    return {
        "section_title": section_title,
        "subsection_title": subsection_title,
        "section_type": section.get("section_type", ""),
        "section_path": section_path,
    }


def _apply_heading(current_section: dict[str, str], heading: dict[str, Any]) -> dict[str, str]:
    title = str(heading["title"])
    section_type = str(heading["section_type"])
    level = int(heading["level"])

    if level > 1 and current_section.get("section_title"):
        next_section = {
            "section_title": current_section.get("section_title", ""),
            "subsection_title": title,
            "section_type": section_type or current_section.get("section_type", ""),
        }
    else:
        next_section = {
            "section_title": title,
            "subsection_title": "",
            "section_type": section_type,
        }
    return _with_section_path(next_section)


def _split_page_into_section_blocks(
    text: str,
    current_section: dict[str, str],
) -> tuple[list[tuple[str, dict[str, str]]], dict[str, str]]:
    blocks: list[tuple[str, dict[str, str]]] = []
    buffer: list[str] = []
    block_section = _with_section_path(current_section)

    def flush() -> None:
        nonlocal buffer
        block_text = "\n".join(buffer).strip()
        if block_text:
            blocks.append((block_text, dict(block_section)))
        buffer = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        heading = _match_section_heading(line)
        if heading:
            flush()
            current_section = _apply_heading(current_section, heading)
            block_section = dict(current_section)
            buffer.append(str(heading["title"]))
            remainder = str(heading.get("remainder") or "").strip()
            if remainder:
                buffer.append(remainder)
        else:
            buffer.append(raw_line)

    flush()
    return blocks, current_section


def split_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[TextChunk]:
    """Split PDF page text into chunks while preserving section metadata."""

    all_chunks: list[TextChunk] = []
    current_section = _empty_section()
    parent_index = 1
    for page in pages:
        section_blocks, current_section = _split_page_into_section_blocks(page["text"], current_section)
        chunk_index = 1
        for block_text, section in section_blocks:
            parent_id = f"{page['paper_id']}_parent_{parent_index}"
            page_chunks = split_text(block_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for child_index, chunk_text in enumerate(page_chunks, start=1):
                chunk_id = f"{page['paper_id']}_page_{page['page']}_chunk_{chunk_index}"
                all_chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        paper_id=page["paper_id"],
                        file_name=page["file_name"],
                        page=int(page["page"]),
                        text=chunk_text,
                        section_title=section.get("section_title", ""),
                        subsection_title=section.get("subsection_title", ""),
                        section_type=section.get("section_type", ""),
                        section_path=section.get("section_path", ""),
                        parent_id=parent_id,
                        parent_index=parent_index,
                        child_index=child_index,
                    )
                )
                chunk_index += 1
            parent_index += 1

    return all_chunks

"""UI-agnostic application services shared by Streamlit and FastAPI."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any, Callable, Iterator
from uuid import uuid4

from backend.agent import ResearchAgent
from backend.config import settings
from backend.conversation_store import ConversationStore
from backend.embeddings import OpenAICompatibleClient
from backend.memory import AgentMemory
from backend.pdf_loader import load_pdf_pages, normalize_paper_id
from backend.prompts import is_long_form_question
from backend.rag_chain import PaperRAG, QueryAnalysis
from backend.summarizer import generate_paper_note
from backend.text_splitter import split_pages
from backend.vector_store import ChromaVectorStore


@dataclass(frozen=True)
class IndexResult:
    paper_id: str
    file_name: str
    page_count: int
    chunk_count: int


class PaperService:
    """Single application boundary for indexing, retrieval, QA, and Agent calls.

    The existing retrieval implementation keeps small in-process caches and the
    latest timing values on the vector-store instance.  A re-entrant lock makes
    those values coherent when FastAPI executes multiple synchronous calls in a
    worker pool.  This is correctness-first; the store can later be made fully
    request-local if higher concurrent throughput is required.
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore | None = None,
        llm_client: OpenAICompatibleClient | None = None,
    ) -> None:
        self.vector_store = vector_store or ChromaVectorStore()
        self._llm_client = llm_client
        self._operation_lock = RLock()

    @property
    def llm_client(self) -> OpenAICompatibleClient:
        if self._llm_client is None:
            self._llm_client = OpenAICompatibleClient()
        return self._llm_client

    def index_pdf(
        self,
        pdf_path: str | Path,
        *,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> IndexResult:
        path = Path(pdf_path)
        paper_id = normalize_paper_id(path.name)
        if progress_callback is not None:
            progress_callback("parsing", 0, 1)
        pages = load_pdf_pages(path, paper_id=paper_id)
        if progress_callback is not None:
            progress_callback("parsing", 1, 1)
            progress_callback("splitting", 0, 1)
        chunks = split_pages(
            pages,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        if not chunks:
            raise ValueError("PDF 中没有可索引的文本内容。")
        if progress_callback is not None:
            progress_callback("splitting", 1, 1)
        with self._operation_lock:
            chunk_count = self.vector_store.index_chunks(
                chunks,
                self.llm_client,
                replace_existing=True,
                progress_callback=progress_callback,
            )
        if progress_callback is not None:
            progress_callback("done", chunk_count, chunk_count)
        return IndexResult(
            paper_id=paper_id,
            file_name=path.name,
            page_count=len(pages),
            chunk_count=chunk_count,
        )

    def list_papers(self) -> list[dict[str, Any]]:
        with self._operation_lock:
            return self.vector_store.list_papers()

    def count_chunks(self) -> int:
        with self._operation_lock:
            return self.vector_store.count_chunks()

    def warmup(self) -> dict[str, Any]:
        with self._operation_lock:
            return self.vector_store.warmup()

    def delete_paper(self, paper_id: str) -> None:
        with self._operation_lock:
            self.vector_store.delete_paper(paper_id)

    def retrieve(
        self,
        question: str,
        *,
        paper_id: str | None = None,
        top_k: int | None = None,
        retrieval_mode: str | None = None,
        expand_parent: bool | None = None,
    ) -> dict[str, Any]:
        resolved_top_k = self._resolve_top_k(paper_id, top_k, question)
        with self._operation_lock:
            hits = self.vector_store.query(
                query_text=question,
                embedding_client=self.llm_client,
                top_k=resolved_top_k,
                paper_id=paper_id,
                retrieval_mode=retrieval_mode,
                expand_parent=expand_parent,
            )
            timings = dict(self.vector_store.last_query_timings_ms)
        return {"hits": hits, "retrieval_timings_ms": timings, "top_k": resolved_top_k}

    def answer(
        self,
        question: str,
        *,
        paper_id: str | None = None,
        top_k: int | None = None,
        retrieval_question: str | None = None,
        retrieval_mode: str | None = None,
        expand_parent: bool | None = None,
        deep_mode: bool | None = None,
    ) -> dict[str, Any]:
        started = perf_counter()
        resolved_deep_mode = PaperRAG.resolve_deep_mode(question, paper_id, deep_mode)
        resolved_top_k = self._resolve_top_k(
            paper_id,
            top_k,
            question,
            deep_mode=resolved_deep_mode,
        )
        with self._operation_lock:
            result = PaperRAG(self.vector_store, self.llm_client).answer(
                question,
                paper_id=paper_id,
                top_k=resolved_top_k,
                retrieval_question=retrieval_question,
                retrieval_mode=retrieval_mode,
                expand_parent=expand_parent,
                deep_mode=resolved_deep_mode,
            )
            result["retrieval_timings_ms"] = dict(self.vector_store.last_query_timings_ms)
        total_ms = (perf_counter() - started) * 1000
        retrieval_ms = float(result["retrieval_timings_ms"].get("total_ms") or 0.0)
        result["timings_ms"] = {
            "retrieval": round(retrieval_ms, 2),
            "generation": round(max(total_ms - retrieval_ms, 0.0), 2),
            "total": round(total_ms, 2),
        }
        result["top_k"] = resolved_top_k
        return result

    def answer_stream(
        self,
        question: str,
        *,
        paper_id: str | None = None,
        top_k: int | None = None,
        retrieval_question: str | None = None,
        retrieval_mode: str | None = None,
        expand_parent: bool | None = None,
        deep_mode: bool | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Retrieve under the store lock, then stream generation without it."""

        started = perf_counter()
        resolved_deep_mode = PaperRAG.resolve_deep_mode(question, paper_id, deep_mode)
        resolved_top_k = self._resolve_top_k(
            paper_id,
            top_k,
            question,
            deep_mode=resolved_deep_mode,
        )
        rag = PaperRAG(self.vector_store, self.llm_client)
        resolved_question = retrieval_question or question
        yield {
            "event": "stage",
            "data": {
                "stage": "retrieving",
                "label": "检索中",
                "message": "正在从论文中查找相关证据…",
            },
        }
        with self._operation_lock:
            hits, retrieval_analysis = rag.retrieve_hits(
                resolved_question,
                paper_id=paper_id,
                top_k=resolved_top_k,
                retrieval_mode=retrieval_mode,
                expand_parent=expand_parent,
                deep_mode=resolved_deep_mode,
            )
            timings = dict(self.vector_store.last_query_timings_ms)
        retrieved_at = perf_counter()

        analysis = QueryAnalysis(
            original_question=question,
            rewritten_query=retrieval_analysis.rewritten_query,
            intent=retrieval_analysis.intent,
            section_types=retrieval_analysis.section_types,
        )

        for event in rag.answer_from_hits_stream(
            question,
            hits,
            analysis=analysis,
            resolved_question=resolved_question if retrieval_question else None,
            deep_mode=resolved_deep_mode,
        ):
            if event["event"] in {"meta", "done"}:
                event["data"]["retrieval_timings_ms"] = timings
                event["data"]["top_k"] = resolved_top_k
            if event["event"] == "done":
                finished_at = perf_counter()
                event["data"]["timings_ms"] = {
                    "retrieval": round((retrieved_at - started) * 1000, 2),
                    "generation": round((finished_at - retrieved_at) * 1000, 2),
                    "total": round((finished_at - started) * 1000, 2),
                }
            yield event

    @staticmethod
    def _resolve_top_k(
        paper_id: str | None,
        top_k: int | None,
        question: str | None = None,
        *,
        deep_mode: bool | None = None,
    ) -> int:
        if top_k is not None:
            return top_k
        resolved_deep_mode = (
            is_long_form_question(question or "")
            if deep_mode is None
            else bool(deep_mode)
        )
        if paper_id and resolved_deep_mode:
            return max(settings.qa_detailed_top_k, settings.retrieval_top_k)
        return settings.retrieval_top_k if paper_id else settings.multi_paper_top_k

    def summarize(self, paper_id: str, *, top_k: int = 12) -> dict[str, Any]:
        with self._operation_lock:
            result = generate_paper_note(paper_id, self.vector_store, self.llm_client, top_k=top_k)
            result["retrieval_timings_ms"] = dict(self.vector_store.last_query_timings_ms)
        return result

    def run_agent(self, prompt: str, memory: AgentMemory) -> dict[str, Any]:
        with self._operation_lock:
            result = ResearchAgent(self.vector_store, self.llm_client, memory).run(prompt)
            timings = dict(self.vector_store.last_query_timings_ms)
        payload = asdict(result)
        payload["retrieval_timings_ms"] = timings
        return payload


@dataclass
class _SessionEntry:
    memory: AgentMemory = field(default_factory=AgentMemory)
    lock: RLock = field(default_factory=RLock)


class AgentSessionStore:
    """Process-local Agent memories isolated by an API session id."""

    def __init__(self, conversation_store: ConversationStore | None = None) -> None:
        self._entries: dict[str, _SessionEntry] = {}
        self._lock = RLock()
        self.conversation_store = conversation_store

    @contextmanager
    def session(self, session_id: str) -> Iterator[AgentMemory]:
        with self._lock:
            entry = self._entries.get(session_id)
            if entry is None:
                memory = (
                    self.conversation_store.load_memory(session_id)
                    if self.conversation_store is not None
                    else AgentMemory()
                )
                entry = _SessionEntry(memory=memory)
                self._entries[session_id] = entry
        with entry.lock:
            try:
                yield entry.memory
            finally:
                if self.conversation_store is not None:
                    self.conversation_store.save_memory(session_id, entry.memory)

    def persist_exchange(
        self,
        session_id: str,
        *,
        user_content: str,
        assistant_content: str,
        intent: str,
        memory: AgentMemory,
    ) -> None:
        if self.conversation_store is None:
            return
        self.conversation_store.save_exchange(
            session_id,
            user_content=user_content,
            assistant_content=assistant_content,
            memory=memory,
            intent=intent,
            rewritten_query=memory.state.rewritten_query,
        )

    def clear(self, session_id: str) -> bool:
        with self._lock:
            removed = self._entries.pop(session_id, None) is not None
        if self.conversation_store is not None and self.conversation_store.conversation_exists(session_id):
            self.conversation_store.delete_conversation(session_id)
            removed = True
        return removed


@dataclass
class TaskSnapshot:
    task_id: str
    status: str = "queued"
    created_at: str = field(default_factory=lambda: _utc_now())
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    progress_stage: str | None = None
    progress_current: int = 0
    progress_total: int = 0
    progress_percent: float = 0.0
    progress_message: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BackgroundTaskManager:
    """Bounded in-process task runner for PDF parsing and embedding jobs."""

    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix="rag-index")
        self._tasks: dict[str, TaskSnapshot] = {}
        self._lock = RLock()

    def submit(
        self,
        operation: Callable[..., dict[str, Any]],
        *,
        with_progress: bool = False,
    ) -> TaskSnapshot:
        task = TaskSnapshot(task_id=uuid4().hex)
        with self._lock:
            self._tasks[task.task_id] = task

        def run() -> None:
            self._update(task.task_id, status="running", started_at=_utc_now())

            def report(stage: str, current: int, total: int, message: str | None = None) -> None:
                bounded_total = max(int(total), 0)
                bounded_current = min(max(int(current), 0), bounded_total) if bounded_total else 0
                percent = (bounded_current / bounded_total * 100) if bounded_total else 0.0
                self._update(
                    task.task_id,
                    progress_stage=stage,
                    progress_current=bounded_current,
                    progress_total=bounded_total,
                    progress_percent=round(percent, 1),
                    progress_message=message,
                )

            try:
                result = operation(report) if with_progress else operation()
            except Exception as exc:  # The task endpoint exposes a safe error string.
                self._update(
                    task.task_id,
                    status="failed",
                    finished_at=_utc_now(),
                    error=f"{type(exc).__name__}: {exc}",
                )
            else:
                self._update(
                    task.task_id,
                    status="succeeded",
                    finished_at=_utc_now(),
                    result=result,
                    progress_stage="done",
                    progress_current=1,
                    progress_total=1,
                    progress_percent=100.0,
                )

        self._executor.submit(run)
        return self.get(task.task_id) or task

    def get(self, task_id: str) -> TaskSnapshot | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return TaskSnapshot(**asdict(task)) if task else None

    def _update(self, task_id: str, **changes: Any) -> None:
        with self._lock:
            task = self._tasks[task_id]
            for name, value in changes.items():
                setattr(task, name, value)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

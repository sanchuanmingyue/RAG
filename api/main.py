"""FastAPI entry point for PaperReader-RAG.

Run with:
    uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
"""

from __future__ import annotations

import os
import json
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import AsyncIterator, Callable
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

from api.schemas import (
    AgentRequest,
    ChatRequest,
    LiteratureSearchRequest,
    MessageResponse,
    RetrieveRequest,
    SummaryRequest,
    TaskResponse,
)
from backend.config import PAPER_DIR, settings
from backend.ieee_search import LiteratureSearchService
from backend.services import AgentSessionStore, BackgroundTaskManager, PaperService


class APIContainer:
    """Long-lived dependencies owned by one API process."""

    def __init__(self) -> None:
        self.paper_service = PaperService()
        self.literature_search_service = LiteratureSearchService(
            llm_client=self.paper_service.llm_client if settings.is_ready else None
        )
        self.sessions = AgentSessionStore()
        self.tasks = BackgroundTaskManager(max_workers=settings.api_background_workers)

    def close(self) -> None:
        self.tasks.close()


def _get_container(request: Request) -> APIContainer:
    return request.app.state.container


def _require_model_configuration() -> None:
    if not settings.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="生成模型或向量模型 API 尚未完整配置，请检查 .env。",
        )


def _require_ieee_configuration() -> None:
    if not settings.ieee_is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="IEEE_API_KEY 尚未配置，请检查 .env。",
        )


def _validate_pdf_name(upload: UploadFile) -> str:
    original = upload.filename or ""
    safe_name = Path(original).name
    if not safe_name or safe_name.lower() != original.lower() or Path(safe_name).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail=f"无效的 PDF 文件名：{original or '(empty)'}")
    return safe_name


async def _persist_upload(upload: UploadFile, safe_name: str) -> Path:
    """Save one bounded upload atomically inside data/papers."""

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    target = (PAPER_DIR / safe_name).resolve()
    paper_root = PAPER_DIR.resolve()
    if target.parent != paper_root:
        raise HTTPException(status_code=400, detail="上传路径无效。")

    temporary = paper_root / f".{uuid4().hex}.uploading"
    max_bytes = max(1, settings.api_max_upload_mb) * 1024 * 1024
    written = 0
    try:
        with temporary.open("wb") as output:
            while chunk := await upload.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"单个 PDF 不能超过 {settings.api_max_upload_mb} MB。",
                    )
                output.write(chunk)
        if written == 0:
            raise HTTPException(status_code=400, detail="上传的 PDF 为空。")
        os.replace(temporary, target)
        return target
    finally:
        await upload.close()
        if temporary.exists():
            temporary.unlink()


def _task_payload(task) -> TaskResponse:
    return TaskResponse(**asdict(task))


def _translate_operation_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ValueError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=503, detail=str(exc))
    return HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


router = APIRouter(prefix="/api/v1")


@router.get("/health")
async def health(container: APIContainer = Depends(_get_container)) -> dict:
    chunk_count = await run_in_threadpool(container.paper_service.count_chunks)
    return {
        "status": "ok",
        "model_configured": settings.is_ready,
        "llm_configured": settings.llm_is_ready,
        "llm_models": settings.llm_models,
        "embedding_configured": settings.embedding_is_ready,
        "embedding_model": settings.embedding_model,
        "ieee_search_configured": settings.ieee_is_ready,
        "single_paper_top_k": settings.retrieval_top_k,
        "multi_paper_top_k": settings.multi_paper_top_k,
        "collection": container.paper_service.vector_store.collection_name,
        "chunk_count": chunk_count,
    }


@router.get("/papers")
async def list_papers(container: APIContainer = Depends(_get_container)) -> dict:
    papers = await run_in_threadpool(container.paper_service.list_papers)
    return {"papers": papers, "count": len(papers)}


@router.post("/papers/index", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def index_papers(
    files: list[UploadFile] = File(...),
    container: APIContainer = Depends(_get_container),
) -> TaskResponse:
    _require_model_configuration()
    if not files:
        raise HTTPException(status_code=400, detail="至少上传一个 PDF。")
    names = [_validate_pdf_name(upload) for upload in files]
    paths = [await _persist_upload(upload, name) for upload, name in zip(files, names)]

    def operation(report_progress) -> dict:
        results = []
        stage_ranges = {
            "parsing": (0.00, 0.15),
            "splitting": (0.15, 0.10),
            "embedding": (0.25, 0.65),
            "writing": (0.90, 0.09),
            "done": (1.00, 0.00),
        }
        for file_index, path in enumerate(paths):
            def on_progress(stage: str, current: int, total: int, *, _index=file_index, _path=path) -> None:
                start, width = stage_ranges.get(stage, (0.0, 0.0))
                ratio = min(max(current / max(total, 1), 0.0), 1.0)
                overall = (_index + min(start + width * ratio, 1.0)) / len(paths)
                report_progress(
                    stage,
                    int(overall * 1000),
                    1000,
                    f"{_index + 1}/{len(paths)} · {_path.name}",
                )

            results.append(asdict(container.paper_service.index_pdf(path, progress_callback=on_progress)))
        return {"papers": results, "total_chunks": sum(item["chunk_count"] for item in results)}

    return _task_payload(container.tasks.submit(operation, with_progress=True))


@router.delete("/papers/{paper_id}", response_model=MessageResponse)
async def delete_paper(paper_id: str, container: APIContainer = Depends(_get_container)) -> MessageResponse:
    await run_in_threadpool(container.paper_service.delete_paper, paper_id)
    return MessageResponse(message=f"已删除论文 {paper_id} 的向量索引；原始 PDF 保留。")


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str, container: APIContainer = Depends(_get_container)) -> TaskResponse:
    task = container.tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在。")
    return _task_payload(task)


@router.post("/retrieve")
async def retrieve(payload: RetrieveRequest, container: APIContainer = Depends(_get_container)) -> dict:
    _require_model_configuration()
    try:
        return await run_in_threadpool(
            container.paper_service.retrieve,
            payload.question,
            paper_id=payload.paper_id,
            top_k=payload.top_k,
            retrieval_mode=payload.retrieval_mode,
            expand_parent=payload.expand_parent,
        )
    except Exception as exc:
        raise _translate_operation_error(exc) from exc


@router.post("/chat")
async def chat(payload: ChatRequest, container: APIContainer = Depends(_get_container)) -> dict:
    _require_model_configuration()
    try:
        return await run_in_threadpool(
            container.paper_service.answer,
            payload.question,
            paper_id=payload.paper_id,
            top_k=payload.top_k,
            retrieval_mode=payload.retrieval_mode,
            expand_parent=payload.expand_parent,
        )
    except Exception as exc:
        raise _translate_operation_error(exc) from exc


@router.post("/literature/search")
async def literature_search(
    payload: LiteratureSearchRequest,
    container: APIContainer = Depends(_get_container),
) -> dict:
    """Search verifiable IEEE Xplore metadata from a natural-language request."""

    _require_ieee_configuration()
    try:
        return await run_in_threadpool(
            container.literature_search_service.search,
            payload.query,
            limit=payload.limit,
        )
    except Exception as exc:
        raise _translate_operation_error(exc) from exc


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
async def chat_stream(payload: ChatRequest, container: APIContainer = Depends(_get_container)) -> StreamingResponse:
    """Stream RAG output as SSE events: meta, delta, done, or error."""

    _require_model_configuration()

    def event_stream():
        try:
            for item in container.paper_service.answer_stream(
                payload.question,
                paper_id=payload.paper_id,
                top_k=payload.top_k,
                retrieval_mode=payload.retrieval_mode,
                expand_parent=payload.expand_parent,
            ):
                yield _sse(item["event"], item["data"])
        except Exception as exc:
            yield _sse("error", {"error": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/summaries")
async def summarize(payload: SummaryRequest, container: APIContainer = Depends(_get_container)) -> dict:
    _require_model_configuration()
    try:
        return await run_in_threadpool(
            container.paper_service.summarize,
            payload.paper_id,
            top_k=payload.top_k,
        )
    except Exception as exc:
        raise _translate_operation_error(exc) from exc


@router.post("/agent")
async def agent(payload: AgentRequest, container: APIContainer = Depends(_get_container)) -> dict:
    _require_model_configuration()
    session_id = payload.session_id or uuid4().hex

    def operation() -> dict:
        with container.sessions.session(session_id) as memory:
            memory.set_scope(payload.current_paper_id, payload.selected_paper_ids)
            result = container.paper_service.run_agent(payload.prompt, memory)
        return {"session_id": session_id, **result}

    try:
        return await run_in_threadpool(operation)
    except Exception as exc:
        raise _translate_operation_error(exc) from exc


@router.delete("/sessions/{session_id}", response_model=MessageResponse)
async def clear_session(session_id: str, container: APIContainer = Depends(_get_container)) -> MessageResponse:
    removed = container.sessions.clear(session_id)
    if not removed:
        raise HTTPException(status_code=404, detail="会话不存在。")
    return MessageResponse(message="会话记忆已清除。")


def create_app(container_factory: Callable[[], APIContainer] = APIContainer) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.container = container_factory()
        warmup = getattr(application.state.container.paper_service, "warmup", None)
        if settings.enable_startup_warmup and warmup is not None:
            application.state.warmup = await run_in_threadpool(warmup)
        try:
            yield
        finally:
            application.state.container.close()

    application = FastAPI(
        title="PaperReader-RAG API",
        version="1.0.0",
        description="论文上传、索引、章节感知检索、问答和 Agent API。",
        lifespan=lifespan,
    )
    application.include_router(router)
    return application


app = create_app()

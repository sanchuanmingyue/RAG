"""Synchronous application boundary for the asynchronous RAG-Anything reader."""

from __future__ import annotations

import asyncio
from pathlib import Path
import threading
from typing import Any, Coroutine, TypeVar

from backend.config import RAG_ANYTHING_DIR
from backend.rag_anything_reader import RagAnythingPaperReader


_T = TypeVar("_T")


def multimodal_index_ready() -> bool:
    """Return whether the persisted LightRAG directory contains a real index."""

    if not RAG_ANYTHING_DIR.exists():
        return False
    ignored = {".gitkeep", "kv_store_llm_response_cache.json"}
    return any(path.is_file() and path.name not in ignored for path in RAG_ANYTHING_DIR.rglob("*"))


class MultimodalPaperService:
    """Keep RAG-Anything on one persistent event loop for Streamlit callers."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="rag-anything-loop",
            daemon=True,
        )
        self._thread.start()
        self._reader = self._run(self._create_reader())

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run(self, coroutine: Coroutine[Any, Any, _T]) -> _T:
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result()

    @staticmethod
    async def _create_reader() -> RagAnythingPaperReader:
        try:
            from lightrag.kg.shared_storage import finalize_share_data

            finalize_share_data()
        except Exception:
            pass
        return RagAnythingPaperReader()

    def parser_ready(self) -> bool:
        async def check() -> bool:
            return self._reader.parser_ready()

        return self._run(check())

    def process_document(self, file_path: str | Path, **kwargs: Any) -> Any:
        return self._run(self._reader.process_document(Path(file_path), **kwargs))

    def ask(self, question: str, *, mode: str = "hybrid") -> Any:
        return self._run(self._reader.ask(question, mode=mode))

    def ask_with_content(
        self,
        question: str,
        multimodal_content: list[dict[str, Any]],
        *,
        mode: str = "hybrid",
    ) -> Any:
        return self._run(
            self._reader.ask_with_content(
                question,
                multimodal_content=multimodal_content,
                mode=mode,
            )
        )

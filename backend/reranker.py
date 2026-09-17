"""Pluggable section reranking through SiliconFlow or a local CrossEncoder."""

from __future__ import annotations

from threading import RLock
from typing import Any

import httpx

from backend.config import settings


def _section_text(hit: dict[str, Any]) -> str:
    metadata = hit.get("metadata", {})
    heading = str(metadata.get("section_path") or metadata.get("section_title") or "")
    text = str(hit.get("section_document") or hit.get("text") or "")
    return f"{heading}\n{text}".strip()[: max(settings.reranker_document_max_chars, 256)]


def _blend_scores(hits: list[dict[str, Any]], raw_scores: list[float], *, provider: str) -> list[dict[str, Any]]:
    if not hits or len(hits) != len(raw_scores):
        return hits
    minimum, maximum = min(raw_scores), max(raw_scores)
    weight = min(max(settings.reranker_weight, 0.0), 1.0)
    section_scores = [float(hit.get("section_score") or hit.get("rerank_score") or 0.0) for hit in hits]
    section_min, section_max = min(section_scores), max(section_scores)
    for hit, raw_score, section_score in zip(hits, raw_scores, section_scores):
        normalized = (raw_score - minimum) / (maximum - minimum) if maximum > minimum else 1.0
        section_normalized = (
            (section_score - section_min) / (section_max - section_min)
            if section_max > section_min
            else 1.0
        )
        hit["section_reranker_provider"] = provider
        hit["section_reranker_score"] = raw_score
        hit["section_reranker_score_normalized"] = normalized
        hit["cross_encoder_score"] = raw_score
        hit["cross_encoder_score_normalized"] = normalized
        hit["final_section_score"] = weight * normalized + (1.0 - weight) * section_normalized
    hits.sort(key=lambda hit: float(hit.get("final_section_score") or 0.0), reverse=True)
    return hits


class SiliconFlowSectionReranker:
    """Call SiliconFlow's native ``/v1/rerank`` endpoint once per query."""

    def __init__(self, client: Any | None = None) -> None:
        self._client = client
        self._error = ""

    @property
    def status(self) -> dict[str, Any]:
        return {
            "provider": "siliconflow",
            "ready": bool(settings.reranker_api_key and settings.reranker_base_url and settings.reranker_model),
            "model": settings.reranker_model,
            "error": self._error,
        }

    def warmup(self) -> bool:
        return bool(settings.reranker_api_key and settings.reranker_base_url and settings.reranker_model)

    def rerank(self, query_text: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.warmup() or not hits:
            return hits
        count = min(len(hits), max(settings.reranker_candidate_k, 1))
        candidates = hits[:count]
        payload = {
            "model": settings.reranker_model,
            "query": query_text,
            "documents": [_section_text(hit) for hit in candidates],
            "top_n": count,
            "return_documents": False,
            "max_chunks_per_doc": max(settings.reranker_max_chunks_per_doc, 1),
            "overlap_tokens": min(max(settings.reranker_overlap_tokens, 0), 80),
        }
        try:
            response = self._get_client().post("/rerank", json=payload)
            response.raise_for_status()
            results = response.json().get("results", [])
            score_by_index = {
                int(item["index"]): float(item["relevance_score"])
                for item in results
                if "index" in item and "relevance_score" in item
            }
            if len(score_by_index) != count:
                raise ValueError(f"rerank response returned {len(score_by_index)}/{count} scores")
            raw_scores = [score_by_index[index] for index in range(count)]
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            return hits
        self._error = ""
        return _blend_scores(candidates, raw_scores, provider="siliconflow") + hits[count:]

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = httpx.Client(
                base_url=settings.reranker_base_url.rstrip("/"),
                headers={"Authorization": f"Bearer {settings.reranker_api_key}"},
                timeout=settings.reranker_timeout_seconds,
            )
        return self._client


class CrossEncoderSectionReranker:
    """Lazy local sentence-transformers backend retained as an offline option."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._error = ""
        self._lock = RLock()

    @property
    def status(self) -> dict[str, Any]:
        return {
            "provider": "local",
            "loaded": self._model is not None,
            "model": settings.cross_encoder_model,
            "error": self._error,
        }

    def warmup(self) -> bool:
        return self._load() is not None

    def rerank(self, query_text: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        model = self._load()
        if model is None or not hits:
            return hits
        count = min(len(hits), max(settings.reranker_candidate_k, 1))
        candidates = hits[:count]
        try:
            raw_scores = [
                float(value)
                for value in model.predict(
                    [(query_text, _section_text(hit)) for hit in candidates],
                    batch_size=settings.cross_encoder_batch_size,
                )
            ]
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            return hits
        return _blend_scores(candidates, raw_scores, provider="local") + hits[count:]

    def _load(self) -> Any | None:
        if self._error:
            return None
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    settings.cross_encoder_model,
                    local_files_only=settings.cross_encoder_local_files_only,
                )
            except Exception as exc:
                self._error = f"{type(exc).__name__}: {exc}"
                return None
        return self._model


class SectionReranker:
    """Select the configured backend while exposing one stable interface."""

    def __init__(self) -> None:
        self.siliconflow = SiliconFlowSectionReranker()
        self.local = CrossEncoderSectionReranker()

    @property
    def status(self) -> dict[str, Any]:
        return {"enabled": settings.enable_section_reranker, **self._backend().status}

    def warmup(self) -> bool:
        if not settings.enable_section_reranker:
            return False
        if settings.reranker_provider == "local" and not settings.preload_cross_encoder:
            return False
        return self._backend().warmup()

    def rerank(self, query_text: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not settings.enable_section_reranker:
            return hits
        return self._backend().rerank(query_text, hits)

    def _backend(self) -> SiliconFlowSectionReranker | CrossEncoderSectionReranker:
        return self.local if settings.reranker_provider == "local" else self.siliconflow

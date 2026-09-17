"""OpenAI-compatible API 封装：embedding 和聊天模型共用同一套配置。"""

from __future__ import annotations

from collections import OrderedDict
import logging
from threading import RLock
import time
from collections.abc import Iterator
from typing import Any, Callable

from openai import OpenAI

from backend.config import settings


logger = logging.getLogger(__name__)

_QUOTA_MARKERS = (
    "allocationquota",
    "free quota exhausted",
    "free tier only",
    "insufficient_quota",
    "quota exceeded",
    "quota exhausted",
    "arrearage",
    "overdue payment",
    "余额不足",
    "额度已用完",
    "额度耗尽",
)
_MODEL_UNAVAILABLE_MARKERS = (
    "model_not_found",
    "model does not exist",
    "model not found",
    "do not have access to it",
    "model unavailable",
)


class OpenAICompatibleClient:
    """封装 OpenAI-compatible API，便于后续替换模型服务。"""

    def __init__(self, *, config=settings, client_factory: Callable[..., Any] = OpenAI) -> None:
        self.config = config
        if not config.is_ready:
            raise RuntimeError(
                "请分别配置 LLM_API_KEY/LLM_BASE_URL/LLM_MODEL 和 "
                "EMBEDDING_API_KEY/EMBEDDING_BASE_URL/EMBEDDING_MODEL"
            )
        self.chat_client = client_factory(api_key=config.llm_api_key, base_url=config.llm_base_url)
        self.embedding_client = client_factory(
            api_key=config.embedding_api_key,
            base_url=config.embedding_base_url,
        )
        # Keep the previous attribute for callers that only used the chat SDK.
        self.client = self.chat_client
        self._query_embedding_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._model_cooldowns: dict[str, float] = {}
        self._model_pool_lock = RLock()
        self.last_chat_metadata: dict[str, Any] = {}

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 10,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        """批量生成文本向量。"""

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.embedding_client.embeddings.create(model=self.config.embedding_model, input=batch)
            embeddings.extend(item.embedding for item in response.data)
            if progress_callback is not None:
                progress_callback(min(start + len(batch), len(texts)), len(texts))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed one user query with a bounded in-process LRU cache.

        Repeated Streamlit submissions, Agent retries, and evaluation reruns often
        ask identical questions.  Caching only queries keeps memory bounded while
        avoiding a network round-trip; document indexing still always embeds the
        supplied chunk content.
        """

        normalized = " ".join(text.split())
        if not normalized:
            raise ValueError("query text must not be empty")

        cached = self._query_embedding_cache.pop(normalized, None)
        if cached is not None:
            self._query_embedding_cache[normalized] = cached
            return cached

        embedding = self.embed_texts([normalized])[0]
        if self.config.query_embedding_cache_size > 0:
            self._query_embedding_cache[normalized] = embedding
            while len(self._query_embedding_cache) > self.config.query_embedding_cache_size:
                self._query_embedding_cache.popitem(last=False)
        return embedding

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        *,
        model_candidates: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Call the ordered model pool and fail over on quota/service errors."""

        attempted: list[str] = []
        skipped: list[str] = []
        last_error: Exception | None = None

        models = model_candidates or self.config.llm_models
        for model in models:
            if self._is_cooling_down(model):
                skipped.append(model)
                continue
            attempted.append(model)
            try:
                response = self.chat_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    **({"max_tokens": max_tokens} if max_tokens and max_tokens > 0 else {}),
                    **self._qwen_thinking_options(model),
                )
                self.last_chat_metadata = {
                    "model_used": model,
                    "attempted_models": attempted.copy(),
                    "skipped_models": skipped.copy(),
                    "fallback_count": max(0, len(attempted) - 1),
                }
                return response.choices[0].message.content or ""
            except Exception as exc:
                failure_kind = self._failover_failure_kind(exc)
                if failure_kind is None:
                    raise
                last_error = exc
                cooldown = (
                    self.config.llm_quota_cooldown_seconds
                    if failure_kind == "quota"
                    else self.config.llm_failover_cooldown_seconds
                )
                self._put_on_cooldown(model, cooldown)
                logger.warning(
                    "LLM model %s failed (%s); trying the next configured model",
                    model,
                    failure_kind,
                )

        self.last_chat_metadata = {
            "model_used": None,
            "attempted_models": attempted,
            "skipped_models": skipped,
            "fallback_count": max(0, len(attempted) - 1),
        }
        if last_error is not None:
            raise RuntimeError(
                f"所有生成模型均不可用，已尝试：{', '.join(attempted)}"
            ) from last_error
        raise RuntimeError(
            "所有生成模型均处于冷却期；请稍后重试或重启服务以清除进程内冷却状态。"
        )

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        *,
        model_candidates: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Stream chat deltas while retaining the ordered model failover policy.

        Failover is safe only before a model emits its first delta. Once output
        has reached the caller, switching models would duplicate or contradict
        the partially generated answer, so later failures are propagated.
        """

        attempted: list[str] = []
        skipped: list[str] = []
        last_error: Exception | None = None

        models = model_candidates or self.config.llm_models
        for model in models:
            if self._is_cooling_down(model):
                skipped.append(model)
                continue
            attempted.append(model)
            emitted = False
            try:
                stream = self.chat_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    stream=True,
                    **({"max_tokens": max_tokens} if max_tokens and max_tokens > 0 else {}),
                    **self._qwen_thinking_options(model),
                )
                for chunk in stream:
                    # DashScope/OpenAI-compatible providers may emit a final
                    # usage/statistics chunk with an empty ``choices`` list.
                    # It is a valid terminator, not a generation failure.
                    choices = getattr(chunk, "choices", None) or []
                    if not choices:
                        continue
                    delta = getattr(choices[0], "delta", None)
                    content = getattr(delta, "content", None) or ""
                    if content:
                        emitted = True
                        yield content
                self.last_chat_metadata = {
                    "model_used": model,
                    "attempted_models": attempted.copy(),
                    "skipped_models": skipped.copy(),
                    "fallback_count": max(0, len(attempted) - 1),
                }
                return
            except Exception as exc:
                if emitted:
                    raise
                failure_kind = self._failover_failure_kind(exc)
                if failure_kind is None:
                    raise
                last_error = exc
                cooldown = (
                    self.config.llm_quota_cooldown_seconds
                    if failure_kind == "quota"
                    else self.config.llm_failover_cooldown_seconds
                )
                self._put_on_cooldown(model, cooldown)
                logger.warning(
                    "Streaming LLM model %s failed (%s); trying the next configured model",
                    model,
                    failure_kind,
                )

        self.last_chat_metadata = {
            "model_used": None,
            "attempted_models": attempted,
            "skipped_models": skipped,
            "fallback_count": max(0, len(attempted) - 1),
        }
        if last_error is not None:
            raise RuntimeError(
                f"所有生成模型均不可用，已尝试：{', '.join(attempted)}"
            ) from last_error
        raise RuntimeError("所有生成模型均处于冷却期；请稍后重试或重启服务以清除进程内冷却状态。")

    def chat_qa(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        """Generate a bounded paper-QA answer without constraining long-form tools."""

        return self.chat(messages, temperature=temperature, max_tokens=self.config.qa_max_tokens)

    def chat_stream_qa(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
    ) -> Iterator[str]:
        """Stream a bounded paper-QA answer."""

        return self.chat_stream(messages, temperature=temperature, max_tokens=self.config.qa_max_tokens)

    def _qwen_thinking_options(self, model: str) -> dict[str, Any]:
        """Apply DashScope's thinking switch only to Qwen model identifiers."""

        if model.strip().lower().startswith("qwen"):
            return {"extra_body": {"enable_thinking": self.config.llm_enable_thinking}}
        return {}

    def _is_cooling_down(self, model: str) -> bool:
        now = time.monotonic()
        with self._model_pool_lock:
            expires_at = self._model_cooldowns.get(model, 0.0)
            if expires_at <= now:
                self._model_cooldowns.pop(model, None)
                return False
            return True

    def _put_on_cooldown(self, model: str, seconds: float) -> None:
        with self._model_pool_lock:
            self._model_cooldowns[model] = time.monotonic() + max(0.0, seconds)

    @staticmethod
    def _failover_failure_kind(exc: Exception) -> str | None:
        """Return a safe failover category, or None for non-retryable requests."""

        status_code = getattr(exc, "status_code", None)
        body = getattr(exc, "body", None)
        error_text = f"{type(exc).__name__} {exc} {body}".lower()

        if any(marker in error_text for marker in _QUOTA_MARKERS):
            return "quota"
        if any(marker in error_text for marker in _MODEL_UNAVAILABLE_MARKERS):
            return "model_unavailable"
        if status_code in {404, 408, 409, 429}:
            return "model_unavailable" if status_code == 404 else "transient"
        if isinstance(status_code, int) and status_code >= 500:
            return "transient"
        if type(exc).__name__ in {"APIConnectionError", "APITimeoutError", "TimeoutException"}:
            return "transient"
        return None

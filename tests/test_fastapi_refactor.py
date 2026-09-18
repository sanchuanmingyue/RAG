"""Network-free checks for the shared service and FastAPI boundary."""

from __future__ import annotations

import asyncio
import os
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import create_app
from backend.config import Settings, settings
from backend.embeddings import OpenAICompatibleClient
from backend.services import AgentSessionStore, BackgroundTaskManager, PaperService
from backend.rag_anything_reader import (
    RagAnythingPaperReader,
    _complete_with_model_pool,
    _ensure_loopback_proxy_bypass,
)


class _FakeVectorStore:
    collection_name = "test_collection"

    def __init__(self) -> None:
        self.last_query_timings_ms: dict[str, float] = {}

    def count_chunks(self) -> int:
        return 3

    def query(self, **kwargs):
        self.last_query_timings_ms = {"total_ms": 12.5}
        return [{"text": "hit", "metadata": {"paper_id": kwargs.get("paper_id") or "p1"}}]


class _FakePaperService:
    def __init__(self) -> None:
        self.vector_store = _FakeVectorStore()

    def list_papers(self):
        return [{"paper_id": "p1", "file_name": "paper.pdf", "chunk_count": 3}]

    def count_chunks(self) -> int:
        return self.vector_store.count_chunks()

    def answer_stream(self, question, **kwargs):
        yield {"event": "meta", "data": {"source_count": 1}}
        yield {"event": "delta", "data": {"text": "答案 [S1]"}}
        yield {"event": "done", "data": {"answer": "答案 [S1]", "sources": []}}


class _FakeLiteratureSearchService:
    def search(self, query, *, limit=10):
        return {
            "request": query,
            "returned": 1,
            "results": [{"title": "Verified IEEE Paper", "rank": 1}],
            "query_plan": {"querytext": "verified query"},
        }


class _FakeContainer:
    def __init__(self) -> None:
        self.paper_service = _FakePaperService()
        self.literature_search_service = _FakeLiteratureSearchService()
        self.sessions = AgentSessionStore()
        self.tasks = BackgroundTaskManager(max_workers=1)

    def close(self) -> None:
        self.tasks.close()


class FastAPIRefactorTests(unittest.TestCase):
    def test_rag_anything_completion_uses_next_configured_model(self) -> None:
        calls: list[str] = []

        async def fake_completion(model, prompt, **kwargs):
            calls.append(model)
            if model == "quota-model":
                raise RuntimeError("Free quota exhausted")
            return "fallback answer"

        with patch.object(settings, "llm_models", ["quota-model", "fallback-model"]):
            result = asyncio.run(
                _complete_with_model_pool(fake_completion, "question")
            )

        self.assertEqual(result, "fallback answer")
        self.assertEqual(calls, ["quota-model", "fallback-model"])

    def test_rag_anything_query_restores_persisted_lightrag_before_query(self) -> None:
        class FakeRag:
            def __init__(self) -> None:
                self.initialized = False

            async def _ensure_lightrag_initialized(self):
                self.initialized = True
                return {"success": True}

            async def aquery(self, question, mode):
                if not self.initialized:
                    raise AssertionError("query ran before persisted storage initialization")
                return "persisted answer"

        reader = RagAnythingPaperReader.__new__(RagAnythingPaperReader)
        reader.rag = FakeRag()

        result = asyncio.run(reader.ask("question", mode="hybrid"))

        self.assertEqual(result, "persisted answer")

    def test_mineru_loopback_requests_bypass_system_proxy(self):
        with patch.dict(
            os.environ,
            {"NO_PROXY": "example.com", "HTTP_PROXY": "", "HTTPS_PROXY": ""},
            clear=False,
        ), patch(
            "backend.rag_anything_reader.urllib.request.getproxies",
            return_value={"http": "http://proxy:7890", "https": "http://proxy:7890"},
        ):
            _ensure_loopback_proxy_bypass()
            self.assertEqual(os.environ["HTTP_PROXY"], "http://proxy:7890")
            self.assertEqual(os.environ["HTTPS_PROXY"], "http://proxy:7890")
            for key in ("NO_PROXY", "no_proxy"):
                values = {item.strip().lower() for item in os.environ[key].split(",")}
                self.assertIn("example.com", values)
                self.assertIn("127.0.0.1", values)
                self.assertIn("localhost", values)
                self.assertIn("::1", values)

    def test_llm_and_embedding_providers_can_be_configured_separately(self) -> None:
        environment = {
            "LLM_API_KEY": "dashscope-key",
            "LLM_BASE_URL": "https://dashscope.example/v1",
            "LLM_MODEL": "glm-5.2",
            "EMBEDDING_API_KEY": "siliconflow-key",
            "EMBEDDING_BASE_URL": "https://api.siliconflow.cn/v1",
            "EMBEDDING_MODEL": "BAAI/bge-m3",
            "VISION_API_KEY": "old-provider-key",
            "VISION_BASE_URL": "https://api.siliconflow.cn/v1",
            "RAG_ANYTHING_VISION_MODEL": "Qwen/Qwen3.5-4B",
        }
        with patch.dict("os.environ", environment, clear=False):
            configured = Settings()

        self.assertTrue(configured.is_ready)
        self.assertEqual(configured.llm_base_url, "https://dashscope.example/v1")
        self.assertEqual(configured.embedding_base_url, "https://api.siliconflow.cn/v1")
        self.assertEqual(configured.embedding_model, "BAAI/bge-m3")
        self.assertEqual(configured.vision_api_key, "siliconflow-key")
        self.assertEqual(configured.rag_anything_vision_model, "Qwen/Qwen3.5-4B")
        self.assertTrue(configured.vision_is_ready)
        self.assertEqual(configured.rag_anything_embedding_dim, 1024)

    def test_llm_reuses_embedding_key_for_the_same_provider(self) -> None:
        environment = {
            "LLM_API_KEY": "",
            "LLM_BASE_URL": "https://api.siliconflow.cn/v1",
            "LLM_MODELS": "THUDM/GLM-4-9B-0414,Qwen/Qwen3.5-4B",
            "EMBEDDING_API_KEY": "shared-siliconflow-key",
            "EMBEDDING_BASE_URL": "https://api.siliconflow.cn/v1",
            "EMBEDDING_MODEL": "BAAI/bge-m3",
        }

        with patch.dict("os.environ", environment, clear=False):
            configured = Settings()

        self.assertTrue(configured.is_ready)
        self.assertEqual(configured.llm_api_key, "shared-siliconflow-key")

    def test_ordered_llm_model_pool_configuration(self) -> None:
        environment = {
            "LLM_API_KEY": "dashscope-key",
            "LLM_BASE_URL": "https://dashscope.example/v1",
            "LLM_MODEL": "legacy-model",
            "LLM_MODELS": "model-a， model-b;model-a； model-c",
            "EMBEDDING_API_KEY": "embedding-key",
            "EMBEDDING_BASE_URL": "https://embedding.example/v1",
            "EMBEDDING_MODEL": "embedding-model",
        }
        with patch.dict("os.environ", environment, clear=False):
            configured = Settings()

        self.assertEqual(configured.llm_models, ["model-a", "model-b", "model-c"])
        self.assertEqual(configured.llm_model, "model-a")

    def test_llm_pool_fails_over_and_keeps_quota_model_on_cooldown(self) -> None:
        class QuotaError(Exception):
            status_code = 400
            body = {"error": {"code": "AllocationQuota.FreeTierOnly"}}

        class FakeCompletions:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def create(self, *, model, **kwargs):
                self.calls.append(model)
                if model == "quota-model":
                    raise QuotaError("Free quota exhausted")
                message = type("Message", (), {"content": "fallback answer"})()
                choice = type("Choice", (), {"message": message})()
                return type("Response", (), {"choices": [choice]})()

        class FakeOpenAI:
            def __init__(self, **kwargs) -> None:
                self.chat = type("Chat", (), {"completions": FakeCompletions()})()
                self.embeddings = object()

        environment = {
            "LLM_API_KEY": "dashscope-key",
            "LLM_BASE_URL": "https://dashscope.example/v1",
            "LLM_MODELS": "quota-model,fallback-model",
            "LLM_QUOTA_COOLDOWN_SECONDS": "86400",
            "EMBEDDING_API_KEY": "embedding-key",
            "EMBEDDING_BASE_URL": "https://embedding.example/v1",
            "EMBEDDING_MODEL": "embedding-model",
        }
        with patch.dict("os.environ", environment, clear=False):
            configured = Settings()
        client = OpenAICompatibleClient(config=configured, client_factory=FakeOpenAI)

        self.assertEqual(client.chat([{"role": "user", "content": "first"}]), "fallback answer")
        self.assertEqual(client.chat([{"role": "user", "content": "second"}]), "fallback answer")
        self.assertEqual(
            client.chat_client.chat.completions.calls,
            ["quota-model", "fallback-model", "fallback-model"],
        )
        self.assertEqual(client.last_chat_metadata["model_used"], "fallback-model")

    def test_qwen_calls_explicitly_disable_thinking_and_stream_deltas(self) -> None:
        class FakeCompletions:
            def __init__(self) -> None:
                self.kwargs = []

            def create(self, **kwargs):
                self.kwargs.append(kwargs)
                if kwargs.get("stream"):
                    delta = type("Delta", (), {"content": "流式答案"})()
                    choice = type("Choice", (), {"delta": delta})()
                    return iter([type("Chunk", (), {"choices": [choice]})()])
                message = type("Message", (), {"content": "普通答案"})()
                choice = type("Choice", (), {"message": message})()
                return type("Response", (), {"choices": [choice]})()

        class FakeOpenAI:
            def __init__(self, **kwargs) -> None:
                self.chat = type("Chat", (), {"completions": FakeCompletions()})()
                self.embeddings = object()

        environment = {
            "LLM_API_KEY": "key", "LLM_BASE_URL": "https://dashscope.example/v1",
            "LLM_MODELS": "qwen3.7-flash", "LLM_ENABLE_THINKING": "false",
            "EMBEDDING_API_KEY": "key", "EMBEDDING_BASE_URL": "https://embedding.example/v1",
            "EMBEDDING_MODEL": "embedding-model",
        }
        with patch.dict("os.environ", environment, clear=False):
            configured = Settings()
        client = OpenAICompatibleClient(config=configured, client_factory=FakeOpenAI)

        self.assertEqual("".join(client.chat_stream([{"role": "user", "content": "hi"}])), "流式答案")
        call = client.chat_client.chat.completions.kwargs[0]
        self.assertTrue(call["stream"])
        self.assertEqual(call["extra_body"], {"enable_thinking": False})

    def test_qa_stream_applies_output_token_limit(self) -> None:
        class FakeCompletions:
            def __init__(self) -> None:
                self.kwargs = []

            def create(self, **kwargs):
                self.kwargs.append(kwargs)
                delta = type("Delta", (), {"content": "bounded"})()
                choice = type("Choice", (), {"delta": delta})()
                return iter([type("Chunk", (), {"choices": [choice]})()])

        class FakeOpenAI:
            def __init__(self, **kwargs) -> None:
                self.chat = type("Chat", (), {"completions": FakeCompletions()})()
                self.embeddings = object()

        environment = {
            "LLM_API_KEY": "key", "LLM_BASE_URL": "https://dashscope.example/v1",
            "LLM_MODELS": "plain-model", "QA_MAX_TOKENS": "321",
            "EMBEDDING_API_KEY": "key", "EMBEDDING_BASE_URL": "https://embedding.example/v1",
            "EMBEDDING_MODEL": "embedding-model",
        }
        with patch.dict("os.environ", environment, clear=False):
            configured = Settings()
        client = OpenAICompatibleClient(config=configured, client_factory=FakeOpenAI)

        self.assertEqual("".join(client.chat_stream_qa([{"role": "user", "content": "hi"}])), "bounded")
        self.assertEqual(client.chat_client.chat.completions.kwargs[0]["max_tokens"], 321)

    def test_chat_stream_ignores_empty_usage_chunks(self) -> None:
        class FakeCompletions:
            def create(self, **kwargs):
                empty = type("Chunk", (), {"choices": []})()
                delta = type("Delta", (), {"content": "完整答案"})()
                choice = type("Choice", (), {"delta": delta})()
                content = type("Chunk", (), {"choices": [choice]})()
                return iter([empty, content, empty])

        class FakeOpenAI:
            def __init__(self, **kwargs) -> None:
                self.chat = type("Chat", (), {"completions": FakeCompletions()})()
                self.embeddings = object()

        environment = {
            "LLM_API_KEY": "key", "LLM_BASE_URL": "https://dashscope.example/v1",
            "LLM_MODELS": "plain-model",
            "EMBEDDING_API_KEY": "key", "EMBEDDING_BASE_URL": "https://embedding.example/v1",
            "EMBEDDING_MODEL": "embedding-model",
        }
        with patch.dict("os.environ", environment, clear=False):
            configured = Settings()
        client = OpenAICompatibleClient(config=configured, client_factory=FakeOpenAI)

        self.assertEqual("".join(client.chat_stream([{"role": "user", "content": "hi"}])), "完整答案")
        self.assertEqual(client.last_chat_metadata["model_used"], "plain-model")

    def test_shared_service_returns_retrieval_timings(self) -> None:
        store = _FakeVectorStore()
        service = PaperService(vector_store=store, llm_client=object())  # type: ignore[arg-type]

        result = service.retrieve("method", paper_id="p1", top_k=4)

        self.assertEqual(result["hits"][0]["text"], "hit")
        self.assertEqual(result["retrieval_timings_ms"]["total_ms"], 12.5)

    def test_service_uses_single_and_multi_paper_top_k_defaults(self) -> None:
        class RecordingStore(_FakeVectorStore):
            def __init__(self) -> None:
                super().__init__()
                self.top_ks = []

            def query(self, **kwargs):
                self.top_ks.append(kwargs["top_k"])
                return super().query(**kwargs)

        store = RecordingStore()
        service = PaperService(vector_store=store, llm_client=object())  # type: ignore[arg-type]

        service.retrieve("single", paper_id="p1")
        service.retrieve("multi")

        self.assertEqual(store.top_ks, [5, 10])

    def test_agent_sessions_are_isolated(self) -> None:
        sessions = AgentSessionStore()
        with sessions.session("a") as memory:
            memory.last_answer = "only-a"
        with sessions.session("b") as memory:
            self.assertEqual(memory.last_answer, "")
        with sessions.session("a") as memory:
            self.assertEqual(memory.last_answer, "only-a")

    def test_background_task_records_result(self) -> None:
        tasks = BackgroundTaskManager(max_workers=1)
        try:
            submitted = tasks.submit(lambda: {"chunk_count": 7})
            deadline = time.monotonic() + 2
            snapshot = tasks.get(submitted.task_id)
            while snapshot and snapshot.status not in {"succeeded", "failed"} and time.monotonic() < deadline:
                time.sleep(0.01)
                snapshot = tasks.get(submitted.task_id)
            self.assertIsNotNone(snapshot)
            self.assertEqual(snapshot.status, "succeeded")
            self.assertEqual(snapshot.result, {"chunk_count": 7})
        finally:
            tasks.close()

    def test_background_task_exposes_index_progress(self) -> None:
        tasks = BackgroundTaskManager(max_workers=1)
        try:
            def operation(report):
                report("embedding", 7, 10, "paper.pdf")
                time.sleep(0.03)
                return {"chunk_count": 10}

            submitted = tasks.submit(operation, with_progress=True)
            deadline = time.monotonic() + 2
            snapshot = tasks.get(submitted.task_id)
            saw_embedding = False
            while snapshot and snapshot.status not in {"succeeded", "failed"} and time.monotonic() < deadline:
                saw_embedding = saw_embedding or snapshot.progress_stage == "embedding"
                time.sleep(0.005)
                snapshot = tasks.get(submitted.task_id)

            self.assertTrue(saw_embedding)
            self.assertIsNotNone(snapshot)
            self.assertEqual(snapshot.progress_percent, 100.0)
            self.assertEqual(snapshot.progress_stage, "done")
        finally:
            tasks.close()

    def test_health_and_paper_routes_start_without_model_calls(self) -> None:
        app = create_app(_FakeContainer)
        with TestClient(app) as client:
            health = client.get("/api/v1/health")
            papers = client.get("/api/v1/papers")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["chunk_count"], 3)
        self.assertEqual(papers.json()["papers"][0]["paper_id"], "p1")

    def test_chat_request_validation_rejects_empty_question(self) -> None:
        app = create_app(_FakeContainer)
        with TestClient(app) as client:
            response = client.post("/api/v1/chat", json={"question": ""})

        self.assertEqual(response.status_code, 422)

    def test_chat_stream_returns_ordered_sse_events(self) -> None:
        app = create_app(_FakeContainer)
        with TestClient(app) as client:
            response = client.post("/api/v1/chat/stream", json={"question": "问题"})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        self.assertLess(response.text.index("event: meta"), response.text.index("event: delta"))
        self.assertLess(response.text.index("event: delta"), response.text.index("event: done"))

    def test_literature_search_endpoint_accepts_natural_language(self) -> None:
        app = create_app(_FakeContainer)
        with patch.object(settings, "ieee_api_key", "test-key"):
            with TestClient(app) as client:
                response = client.post(
                    "/api/v1/literature/search",
                    json={"query": "查找多模态 RAG 论文", "limit": 5},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["results"][0]["title"], "Verified IEEE Paper")

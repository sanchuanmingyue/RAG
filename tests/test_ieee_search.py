"""Network-free tests for natural-language IEEE literature search."""

from __future__ import annotations

import unittest

import httpx

from backend.ieee_search import (
    IEEEXploreClient,
    IEEEQueryPlanner,
    SearchPlan,
    is_literature_search_query,
)


class _Config:
    ieee_api_key = "test-key"
    ieee_api_base_url = "https://example.test/api/v1"
    ieee_search_timeout_seconds = 2.0
    ieee_is_ready = True


class _FakeLLM:
    def chat(self, messages, **kwargs):
        return '{"querytext":"multimodal RAG AND scientific documents","start_year":2022,"end_year":2025,"explanation":"聚焦主题"}'


class IEEESearchTests(unittest.TestCase):
    def test_detects_explicit_literature_search_without_hijacking_local_qa(self) -> None:
        self.assertTrue(is_literature_search_query("帮我查找近三年关于多模态 RAG 的 IEEE 论文"))
        self.assertTrue(is_literature_search_query("find papers about retrieval augmented generation"))
        self.assertFalse(is_literature_search_query("这几篇论文分别采用了什么方法"))

    def test_planner_translates_query_and_keeps_year_range(self) -> None:
        plan = IEEEQueryPlanner(_FakeLLM()).plan("查找 2022 到 2025 年的多模态 RAG 论文")
        self.assertEqual(plan.querytext, "multimodal RAG AND scientific documents")
        self.assertEqual((plan.start_year, plan.end_year), (2022, 2025))

    def test_normalizes_filters_deduplicates_and_scores_results(self) -> None:
        payload = {
            "total_records": 3,
            "articles": [
                {
                    "title": "Multimodal Retrieval Augmented Generation for Documents",
                    "authors": {"authors": [{"full_name": "A. Researcher"}]},
                    "publication_title": "IEEE Access",
                    "publication_year": "2024",
                    "doi": "10.1109/ACCESS.2024.1",
                    "article_number": "123",
                    "abstract": "A multimodal retrieval augmented generation system.",
                },
                {
                    "title": "Multimodal Retrieval Augmented Generation for Documents",
                    "publication_year": "2024",
                    "doi": "10.1109/ACCESS.2024.1",
                },
                {
                    "title": "An Old Paper",
                    "publication_year": "2018",
                    "article_number": "999",
                },
            ],
        }

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.params["apikey"], "test-key")
            self.assertNotIn("sort_field", request.url.params)
            return httpx.Response(200, json=payload)

        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        try:
            result = IEEEXploreClient(config=_Config(), http_client=http_client).search(
                SearchPlan("multimodal retrieval augmented generation", 2022, 2025),
                limit=10,
            )
        finally:
            http_client.close()

        self.assertEqual(result["returned"], 1)
        paper = result["results"][0]
        self.assertEqual(paper["authors"], ["A. Researcher"])
        self.assertEqual(paper["ieee_url"], "https://ieeexplore.ieee.org/document/123")
        self.assertGreater(paper["score"]["total"], 0)
        self.assertLessEqual(paper["score"]["total"], 12)


if __name__ == "__main__":
    unittest.main()

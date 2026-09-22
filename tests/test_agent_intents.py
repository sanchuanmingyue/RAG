"""Intent routing and deterministic Agent tool checks."""

from __future__ import annotations

import unittest

from backend.corpus_analysis import CorpusAnalysisService
from backend.router import AgentRouter, is_library_status_query
from backend.tools import LibraryStatusTool, LiteratureSearchTool


class _Store:
    def list_papers(self):
        return [
            {"paper_id": "p1", "file_name": "paper-a.pdf", "chunk_count": 12},
            {"paper_id": "p2", "file_name": "paper-b.pdf", "chunk_count": 8},
        ]

    def count_chunks(self):
        return 20


class _SearchService:
    def search(self, request):
        return {
            "request": request,
            "returned": 1,
            "query_plan": {"querytext": "multimodal RAG"},
            "results": [
                {
                    "rank": 1,
                    "title": "Verified Paper",
                    "authors": ["A. Author"],
                    "venue": "IEEE Access",
                    "year": 2025,
                    "doi": "10.1109/example",
                    "ieee_url": "https://ieeexplore.ieee.org/document/1",
                    "score": {"total": 10.5},
                }
            ],
        }


class _ProfileStore:
    def get_corpus_background_profiles(self):
        return [
            {"paper_id": "a1", "file_name": "medical-1.pdf", "text": "clinical medical diagnosis patient", "embedding": [1.0, 0.0]},
            {"paper_id": "a2", "file_name": "medical-2.pdf", "text": "biomedical patient question answering", "embedding": [0.95, 0.05]},
            {"paper_id": "a3", "file_name": "medical-3.pdf", "text": "clinical decision support", "embedding": [0.9, 0.1]},
            {"paper_id": "b1", "file_name": "network-1.pdf", "text": "wireless network edge computing", "embedding": [0.0, 1.0]},
            {"paper_id": "b2", "file_name": "network-2.pdf", "text": "mobile communication network", "embedding": [0.05, 0.95]},
            {"paper_id": "b3", "file_name": "network-3.pdf", "text": "edge resource allocation", "embedding": [0.1, 0.9]},
        ]


class _CategoryLLM:
    def chat(self, messages, **kwargs):
        return '{"0":{"label":"类别甲","description":"第一类"},"1":{"label":"类别乙","description":"第二类"}}'


class AgentIntentTests(unittest.TestCase):
    def test_routes_library_status_before_default_qa(self) -> None:
        router = AgentRouter()
        self.assertEqual(router.route("当前库中有多少篇文章"), "library_status")
        self.assertEqual(router.route("列出知识库里已索引的文件"), "library_status")
        self.assertTrue(is_library_status_query("向量库现在有多少 chunks"))
        self.assertEqual(router.route("List the files in the current collection"), "library_status")

    def test_routes_english_summary_and_corpus_analysis(self) -> None:
        router = AgentRouter()
        self.assertEqual(router.route("Summarize the main contribution of this paper"), "summary")
        self.assertEqual(
            router.route("Cluster all papers in the library by research topic"),
            "corpus_analysis",
        )
        self.assertEqual(router.route("这篇论文有多少组实验？"), "qa")

    def test_routes_multi_paper_explanation_to_comparison_pipeline(self) -> None:
        router = AgentRouter()
        self.assertEqual(router.route("这些文章分别在做什么"), "compare")
        self.assertEqual(router.route("这几篇论文各自的方法和实验是什么"), "compare")

    def test_routes_external_literature_search(self) -> None:
        self.assertEqual(
            AgentRouter().route("帮我查找近三年的多模态 RAG IEEE 论文"),
            "literature_search",
        )

    def test_routes_full_library_classification(self) -> None:
        router = AgentRouter()
        self.assertEqual(router.route("对这83篇文章按研究背景分类"), "corpus_analysis")
        self.assertEqual(router.route("把知识库中所有论文分成5类"), "corpus_analysis")
        self.assertEqual(router.route("这篇论文使用了什么分类方法"), "qa")

    def test_library_status_tool_returns_exact_counts_without_llm(self) -> None:
        result = LibraryStatusTool(_Store()).run("列出知识库中的论文和 chunks")
        self.assertEqual(result.intent, "library_status")
        self.assertIn("2 篇", result.answer)
        self.assertIn("20 个 chunks", result.answer)
        self.assertIn("paper-a.pdf", result.answer)
        self.assertEqual(result.artifacts["paper_count"], 2)

    def test_literature_search_tool_keeps_verified_metadata(self) -> None:
        result = LiteratureSearchTool(_SearchService()).run("查找多模态 RAG 论文")
        self.assertEqual(result.intent, "literature_search")
        self.assertIn("Verified Paper", result.answer)
        self.assertIn("10.1109/example", result.answer)
        self.assertEqual(result.artifacts["search"]["returned"], 1)

    def test_corpus_analysis_classifies_every_profile(self) -> None:
        payload = CorpusAnalysisService(_ProfileStore(), _CategoryLLM()).classify_by_background(
            "把全部论文按研究背景分成2类"
        )
        self.assertEqual(payload["paper_count"], 6)
        self.assertEqual(payload["category_count"], 2)
        self.assertEqual(sum(item["count"] for item in payload["categories"]), 6)
        self.assertEqual({item["label"] for item in payload["categories"]}, {"类别甲", "类别乙"})


if __name__ == "__main__":
    unittest.main()

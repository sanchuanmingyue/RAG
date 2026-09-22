"""Network-free checks for the arXiv MCP adapter."""

from __future__ import annotations

import unittest

from backend.arxiv_mcp import ArxivMCPClient, ArxivSearchService, choose_literature_source
from backend.ieee_search import IEEEQueryPlanner
from backend.tools import ArxivSearchTool


class _Planner:
    def plan(self, request):
        from backend.ieee_search import SearchPlan

        return SearchPlan("retrieval augmented generation", 2023, 2026, "test")


class _MCPClient:
    def __init__(self) -> None:
        self.calls = []

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return {
            "total_results": 42,
            "papers": [
                {
                    "id": "2401.00001",
                    "title": "Grounded Retrieval",
                    "authors": ["A. Author"],
                    "abstract": "A grounded RAG paper.",
                    "categories": ["cs.IR"],
                    "published": "2024-01-01T00:00:00Z",
                    "url": "https://arxiv.org/pdf/2401.00001",
                    "resource_uri": "arxiv://2401.00001",
                }
            ],
        }


class ArxivMCPTests(unittest.TestCase):
    def test_explicit_arxiv_request_selects_mcp(self) -> None:
        self.assertEqual(choose_literature_source("帮我搜索 arXiv 上的 RAG 论文"), "arxiv")

    def test_english_search_instruction_is_removed_from_query(self) -> None:
        plan = IEEEQueryPlanner(provider="arXiv").plan(
            "search arXiv papers about retrieval augmented generation"
        )
        self.assertEqual(plan.querytext, "retrieval augmented generation")
        self.assertEqual(choose_literature_source("查找 IEEE RAG 论文"), "ieee")

    def test_search_normalizes_mcp_payload_and_records_tool_call(self) -> None:
        client = _MCPClient()
        service = ArxivSearchService(mcp_client=client)
        service.planner = _Planner()

        payload = service.search("查找近年的 RAG 预印本", limit=5)

        self.assertEqual(client.calls[0][0], "search_papers")
        self.assertEqual(client.calls[0][1]["date_from"], "2023-01-01")
        self.assertEqual(payload["source"], "arXiv MCP")
        self.assertEqual(payload["total_records"], 42)
        self.assertEqual(payload["results"][0]["arxiv_id"], "2401.00001")
        self.assertEqual(payload["mcp"]["tool"], "search_papers")

    def test_arxiv_tool_formats_papers(self) -> None:
        client = _MCPClient()
        service = ArxivSearchService(mcp_client=client)
        service.planner = _Planner()

        result = ArxivSearchTool(service).run("搜索 arXiv RAG 论文")

        self.assertIn("Grounded Retrieval", result.answer)
        self.assertIn("2401.00001", result.answer)
        self.assertEqual(result.artifacts["provider"], "arxiv_mcp")

    def test_mcp_client_rejects_unapproved_tools(self) -> None:
        with self.assertRaises(ValueError):
            ArxivMCPClient().call_tool("delete_everything", {})


if __name__ == "__main__":
    unittest.main()

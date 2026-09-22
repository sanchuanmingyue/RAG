"""LangGraph routing and bounded evidence-retry checks."""

from __future__ import annotations

import unittest

from backend.agent import ResearchAgent
from backend.evaluator import ResultEvaluator
from backend.memory import AgentMemory
from backend.schemas import AgentResult


class _Tool:
    def __init__(self, result=None, *, fail=False) -> None:
        self.result = result
        self.fail = fail
        self.calls = []

    def run(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.fail:
            raise AssertionError("unexpected tool call")
        return self.result


class _RetryQATool:
    def __init__(self) -> None:
        self.calls = []

    def run(self, question, paper_id=None, *, retrieval_question=None):
        self.calls.append(retrieval_question)
        if len(self.calls) == 1:
            return AgentResult(answer="论文中没有找到明确依据。", intent="qa", sources=[])
        return AgentResult(
            answer="方法使用章节级检索 [S1]。",
            intent="qa",
            sources=[{"text": "章节级检索", "metadata": {"paper_id": "p1"}}],
        )


class _LLM:
    def chat(self, messages, **kwargs):
        return "section-level retrieval method architecture"


def _agent() -> ResearchAgent:
    agent = ResearchAgent.__new__(ResearchAgent)
    agent.memory = AgentMemory(current_paper_id="p1")
    agent.llm_client = _LLM()
    agent.evaluator = ResultEvaluator()
    agent.qa_tool = _RetryQATool()
    agent.summary_tool = _Tool(fail=True)
    agent.compare_tool = _Tool(fail=True)
    agent.source_tool = _Tool(fail=True)
    agent.export_tool = _Tool(fail=True)
    agent.library_status_tool = _Tool(fail=True)
    agent.literature_search_tool = _Tool(fail=True)
    agent.arxiv_search_tool = _Tool(
        AgentResult(
            answer="arXiv result",
            intent="literature_search",
            artifacts={"search": {"source": "arXiv MCP"}},
        )
    )
    agent.corpus_classification_tool = _Tool(fail=True)
    agent.graph = agent._build_graph()
    return agent


class AgentGraphTests(unittest.TestCase):
    def test_qa_rewrites_and_retries_at_most_once(self) -> None:
        agent = _agent()

        result = agent.run("论文的核心方法是什么？")

        self.assertEqual(len(agent.qa_tool.calls), 2)
        self.assertEqual(agent.qa_tool.calls[1], "section-level retrieval method architecture")
        self.assertEqual(result.artifacts["workflow"]["engine"], "langgraph")
        self.assertEqual(result.artifacts["workflow"]["retry_count"], 1)
        nodes = [item["node"] for item in result.artifacts["workflow"]["trace"]]
        self.assertEqual(nodes.count("evidence_gate"), 2)
        self.assertIn("rewrite_query", nodes)
        self.assertEqual(len(agent.memory.history), 2)

    def test_explicit_arxiv_search_uses_mcp_branch(self) -> None:
        agent = _agent()

        result = agent.run("帮我在 arXiv 搜索 RAG 论文")

        self.assertEqual(result.answer, "arXiv result")
        self.assertEqual(len(agent.arxiv_search_tool.calls), 1)
        trace = result.artifacts["workflow"]["trace"]
        external = next(item for item in trace if item["node"] == "external_search")
        self.assertEqual(external["provider"], "arxiv")


if __name__ == "__main__":
    unittest.main()

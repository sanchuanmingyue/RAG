"""Regression tests for context-aware multi-turn routing and retrieval rewriting."""

from __future__ import annotations

import unittest

from backend.conversation import ConversationResolver, FollowUpResolution
from backend.memory import AgentMemory, ConversationState
from backend.rag_chain import PaperRAG
from backend.tools import CorpusClassificationTool


def _classification_artifact() -> dict:
    return {
        "artifact_id": "corpus_cls_old",
        "paper_count": 6,
        "category_count": 2,
        "method": "section embeddings + KMeans",
        "silhouette_score": 0.3,
        "categories": [
            {
                "rank": 1,
                "label": "医学问答",
                "count": 3,
                "keywords": ["medical", "patient"],
                "representative_papers": ["a.pdf"],
                "papers": [
                    {"paper_id": "a", "file_name": "a.pdf"},
                    {"paper_id": "b", "file_name": "b.pdf"},
                    {"paper_id": "c", "file_name": "c.pdf"},
                ],
            },
            {
                "rank": 2,
                "label": "边缘计算",
                "count": 3,
                "keywords": ["edge", "offloading"],
                "representative_papers": ["d.pdf"],
                "papers": [
                    {"paper_id": "d", "file_name": "d.pdf"},
                    {"paper_id": "e", "file_name": "e.pdf"},
                    {"paper_id": "f", "file_name": "f.pdf"},
                ],
            },
        ],
    }


class _RewriteLLM:
    def chat(self, messages, **kwargs):
        return "请详细解释论文中双荷兰拍卖机制的实现流程"


class _RecordingStore:
    def __init__(self):
        self.query_text = ""

    def query(self, *, query_text, **kwargs):
        self.query_text = query_text
        return [
            {
                "text": "The auction updates bids and matches buyers with sellers.",
                "metadata": {"file_name": "paper.pdf", "page": 3, "chunk_id": "c1"},
            }
        ]


class _AnswerLLM:
    def __init__(self):
        self.messages = []
        self.last_chat_metadata = {"model_used": "fake"}

    def chat_qa(self, messages, **kwargs):
        self.messages = messages
        return "系统通过更新报价并匹配买卖双方完成拍卖。[S1]"

    def chat(self, messages, **kwargs):
        return self.chat_qa(messages, **kwargs)


class _CorpusService:
    def __init__(self):
        self.category_count = None

    def classify_by_background(self, request, **kwargs):
        self.category_count = kwargs.get("category_count")
        return {
            "artifact_id": "corpus_cls_new",
            "paper_count": 6,
            "category_count": self.category_count,
            "silhouette_score": 0.4,
            "method": "test",
            "categories": [],
        }


class ConversationFollowupTests(unittest.TestCase):
    def test_classification_detail_followup_inherits_corpus_intent(self) -> None:
        state = ConversationState(
            last_intent="corpus_analysis",
            last_user_query="对这83篇论文按研究背景分类",
            active_artifact_type="corpus_analysis",
            active_artifact=_classification_artifact(),
        )
        resolution = ConversationResolver().resolve("能不能再分类详细一点", state)
        self.assertTrue(resolution.is_followup)
        self.assertEqual(resolution.intent, "corpus_analysis")
        self.assertEqual(resolution.operation, "refine_classification")

    def test_category_list_is_an_artifact_operation(self) -> None:
        state = ConversationState(
            last_intent="corpus_analysis",
            active_artifact_type="corpus_analysis",
            active_artifact=_classification_artifact(),
        )
        resolution = ConversationResolver().resolve("列出第二类论文", state)
        self.assertEqual(resolution.operation, "list_category")
        self.assertEqual(resolution.target_category, 2)

        tool = CorpusClassificationTool(object(), object())
        result = tool.run(
            "列出第二类论文",
            previous_artifact=state.active_artifact,
            resolution=resolution,
        )
        self.assertIn("d.pdf", result.answer)
        self.assertIn("f.pdf", result.answer)

    def test_explicit_search_wins_over_previous_corpus_intent(self) -> None:
        state = ConversationState(last_intent="corpus_analysis", last_user_query="分类论文")
        resolution = ConversationResolver().resolve("继续搜索2026年的其他论文", state)
        self.assertEqual(resolution.intent, "literature_search")

    def test_qa_followup_is_rewritten_for_retrieval_but_keeps_original_prompt(self) -> None:
        state = ConversationState(
            last_intent="qa",
            last_user_query="论文为什么使用双荷兰拍卖？",
            last_answer_summary="论文使用双荷兰拍卖进行资源交易。",
        )
        resolution = ConversationResolver(_RewriteLLM()).resolve("具体是怎么实现的？", state)
        self.assertEqual(
            resolution.rewritten_query,
            "请详细解释论文中双荷兰拍卖机制的实现流程",
        )

        store = _RecordingStore()
        llm = _AnswerLLM()
        result = PaperRAG(store, llm).answer(
            "具体是怎么实现的？",
            retrieval_question=resolution.rewritten_query,
        )
        self.assertIn("双荷兰拍卖", store.query_text)
        prompt = llm.messages[-1]["content"]
        self.assertIn("当前用户追问：具体是怎么实现的？", prompt)
        self.assertIn("双荷兰拍卖机制", prompt)
        self.assertIsNone(result["refusal_reason"])

    def test_memory_keeps_compact_state_and_structured_artifact(self) -> None:
        memory = AgentMemory()
        artifact = _classification_artifact()
        memory.remember_result(
            "分类完成",
            [],
            intent="corpus_analysis",
            user_query="对论文分类",
            artifacts={"corpus_analysis": artifact},
        )
        self.assertEqual(memory.state.last_intent, "corpus_analysis")
        self.assertEqual(memory.state.active_artifact_id, "corpus_cls_old")
        self.assertIn("医学问答", memory.state.last_answer_summary)

    def test_refine_operation_increases_category_count_without_losing_context(self) -> None:
        tool = CorpusClassificationTool(object(), object())
        service = _CorpusService()
        tool.service = service
        resolution = FollowUpResolution(
            True,
            "corpus_analysis",
            "artifact_operation",
            "refine_classification",
        )
        result = tool.run(
            "再分类详细一点",
            previous_artifact=_classification_artifact(),
            resolution=resolution,
        )
        self.assertEqual(service.category_count, 4)
        self.assertEqual(result.artifacts["corpus_analysis"]["category_count"], 4)


if __name__ == "__main__":
    unittest.main()

"""Small, network-free checks for retrieval performance/ranking helpers."""

from __future__ import annotations

import unittest
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
import time

from backend.reranker import CrossEncoderSectionReranker, SiliconFlowSectionReranker
from backend.text_splitter import TextChunk
from backend.config import settings
from backend.prompts import build_context
from backend.vector_store import ChromaVectorStore, _tokenize


class RetrievalOptimizationTests(unittest.TestCase):
    def test_query_timings_are_isolated_per_retrieval_thread(self) -> None:
        store = ChromaVectorStore.__new__(ChromaVectorStore)
        store.last_query_timings_ms = {}

        def record(value: float) -> float:
            store.last_query_timings_ms = {"total_ms": value}
            time.sleep(0.01)
            return store.last_query_timings_ms["total_ms"]

        with ThreadPoolExecutor(max_workers=3) as executor:
            recorded = list(executor.map(record, [1.0, 2.0, 3.0]))

        self.assertEqual(recorded, [1.0, 2.0, 3.0])
        self.assertEqual(store.last_query_timings_ms, {})

    def test_section_index_uses_mean_chunk_embedding(self) -> None:
        captured = {}

        class FakeSectionCollection:
            def get(self, **kwargs):
                return {"ids": [], "documents": [], "metadatas": [], "embeddings": []}

            def upsert(self, **kwargs):
                captured.update(kwargs)

        store = ChromaVectorStore.__new__(ChromaVectorStore)
        store.section_collection = FakeSectionCollection()
        chunks = [
            TextChunk("c1", "p1", "p.pdf", 1, "first", section_path="2 Method", parent_id="s1"),
            TextChunk("c2", "p1", "p.pdf", 1, "second", section_path="2 Method", parent_id="s1"),
        ]

        store._upsert_section_records(chunks, [[1.0, 3.0], [3.0, 1.0]])

        self.assertEqual(captured["embeddings"], [[2.0, 2.0]])
        self.assertEqual(captured["metadatas"][0]["section_chunk_count"], 2)
        self.assertIn("first", captured["documents"][0])
        self.assertIn("second", captured["documents"][0])

    def test_keyword_query_reuses_unfiltered_query_scores(self) -> None:
        store = ChromaVectorStore.__new__(ChromaVectorStore)
        store._keyword_rows = [
            type("Row", (), {"text": "retrieval", "metadata": {"paper_id": "p1"}})(),
            type("Row", (), {"text": "retrieval", "metadata": {"paper_id": "p2"}})(),
        ]
        store._keyword_postings = {"retrieval": [(0, 1), (1, 1)]}
        store._keyword_doc_frequency = Counter({"retrieval": 2})
        store._keyword_query_cache = OrderedDict()
        store._last_keyword_timings_ms = {}

        first = store.keyword_query("retrieval", paper_ids=["p1"])
        second = store.keyword_query("retrieval", paper_ids=["p2"])

        self.assertEqual(first[0]["metadata"]["paper_id"], "p1")
        self.assertEqual(second[0]["metadata"]["paper_id"], "p2")
        self.assertEqual(store._last_keyword_timings_ms["cache_hits"], 1.0)

    def test_cross_encoder_blends_and_reorders_section_candidates(self) -> None:
        class FakeModel:
            def predict(self, pairs, batch_size):
                return [0.1, 0.9]

        reranker = CrossEncoderSectionReranker()
        reranker._model = FakeModel()
        hits = [
            {"text": "first", "metadata": {}, "section_score": 0.9},
            {"text": "second", "metadata": {}, "section_score": 0.8},
        ]

        ranked = reranker.rerank("question", hits)

        self.assertEqual(ranked[0]["text"], "second")
        self.assertIn("cross_encoder_score_normalized", ranked[0])

    def test_siliconflow_reranker_uses_native_api_and_preserves_local_fallback(self) -> None:
        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "results": [
                        {"index": 1, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.1},
                    ]
                }

        class FakeClient:
            def post(self, path, json):
                captured["path"] = path
                captured["json"] = json
                return FakeResponse()

        reranker = SiliconFlowSectionReranker(client=FakeClient())
        hits = [
            {"text": "first", "metadata": {}, "section_score": 0.9},
            {"text": "second", "metadata": {}, "section_score": 0.8},
        ]

        old_api_key = settings.reranker_api_key
        settings.reranker_api_key = "test-key"
        try:
            ranked = reranker.rerank("question", hits)
        finally:
            settings.reranker_api_key = old_api_key

        self.assertEqual(captured["path"], "/rerank")
        self.assertEqual(captured["json"]["model"], "BAAI/bge-reranker-v2-m3")
        self.assertEqual(ranked[0]["text"], "second")
        self.assertEqual(ranked[0]["section_reranker_provider"], "siliconflow")

    def test_embedding_text_preserves_chunk_body_and_includes_section_path(self) -> None:
        chunk = TextChunk(
            chunk_id="chunk-1",
            paper_id="paper-1",
            file_name="paper.pdf",
            page=1,
            text="The ablation uses a fixed learning rate.",
            section_path="3 Experiments > 3.2 Ablation",
        )

        embedding_text = chunk.to_embedding_text()

        self.assertTrue(embedding_text.endswith(chunk.text))
        self.assertIn("Document: paper.pdf", embedding_text)
        self.assertIn("Section: 3 Experiments > 3.2 Ablation", embedding_text)
        self.assertEqual(chunk.to_embedding_text(False), chunk.text)

    def test_tokenize_adds_chinese_character_and_bigram_terms(self) -> None:
        tokens = _tokenize("检索性能 RAG")

        self.assertTrue({"检", "索", "检索", "索性", "性能", "rag"}.issubset(tokens))

    def test_section_diversity_defers_excess_hits_from_one_section(self) -> None:
        hits = [
            {"text": "a", "metadata": {"paper_id": "p", "parent_id": "s1", "section_path": "1 Intro"}},
            {"text": "b", "metadata": {"paper_id": "p", "parent_id": "s1", "section_path": "1 Intro"}},
            {"text": "c", "metadata": {"paper_id": "p", "parent_id": "s1", "section_path": "1 Intro"}},
            {"text": "d", "metadata": {"paper_id": "p", "parent_id": "s2", "section_path": "2 Method"}},
        ]

        diversified = ChromaVectorStore._diversify_sections(hits, max_per_section=2)

        self.assertEqual([hit["text"] for hit in diversified], ["a", "b", "d", "c"])

    def test_rerank_normalizes_keyword_score_within_query(self) -> None:
        hits = [
            {"text": "retrieval", "keyword_score": 100.0, "metadata": {}},
            {"text": "retrieval", "keyword_score": 50.0, "metadata": {}},
        ]

        reranked = ChromaVectorStore._rerank(ChromaVectorStore.__new__(ChromaVectorStore), "retrieval", hits)

        self.assertEqual(reranked[0]["keyword_score_normalized"], 1.0)
        self.assertEqual(reranked[1]["keyword_score_normalized"], 0.5)
        self.assertLess(reranked[0]["rerank_score"], 2.0)

    def test_reference_section_is_penalized_unless_explicitly_requested(self) -> None:
        store = ChromaVectorStore.__new__(ChromaVectorStore)
        ordinary = [{"text": "method details", "metadata": {"section_type": "references"}}]
        requested = [{"text": "citation details", "metadata": {"section_type": "references"}}]

        store._rerank("method details", ordinary)
        store._rerank("Which references are cited?", requested)

        self.assertGreater(ordinary[0]["reference_penalty_applied"], 0)
        self.assertEqual(requested[0]["reference_penalty_applied"], 0)

    def test_section_aggregation_returns_one_result_per_section(self) -> None:
        hits = [
            {
                "text": "retrieval method",
                "metadata": {"paper_id": "p1", "parent_id": "s1", "chunk_id": "c1"},
                "rerank_score": 0.9,
                "vector_score_normalized": 0.8,
                "keyword_score_normalized": 0.6,
            },
            {
                "text": "dense retrieval",
                "metadata": {"paper_id": "p1", "parent_id": "s1", "chunk_id": "c2"},
                "rerank_score": 0.7,
                "vector_score_normalized": 0.7,
                "keyword_score_normalized": 0.7,
            },
            {
                "text": "unrelated conclusion",
                "metadata": {"paper_id": "p1", "parent_id": "s2", "chunk_id": "c3"},
                "rerank_score": 0.2,
                "vector_score_normalized": 0.2,
                "keyword_score_normalized": 0.0,
            },
        ]

        sections = ChromaVectorStore._aggregate_sections("retrieval method", hits, support_chunks=3)

        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0]["metadata"]["parent_id"], "s1")
        self.assertEqual(sections[0]["section_chunk_count"], 2)
        self.assertEqual(sections[0]["section_supporting_chunk_ids"], ["c1", "c2"])
        self.assertEqual(sections[0]["generation_text"], "retrieval method\n\ndense retrieval")

    def test_generation_context_uses_supporting_chunks_with_character_budgets(self) -> None:
        old_source_max = settings.qa_source_max_chars
        old_context_max = settings.qa_context_max_chars
        settings.qa_source_max_chars = 12
        settings.qa_context_max_chars = 20
        try:
            context = build_context(
                [
                    {
                        "text": "representative only",
                        "generation_text": "relevant-A relevant-B",
                        "metadata": {"file_name": "a.pdf", "section_path": "Method"},
                    },
                    {
                        "text": "fallback evidence",
                        "metadata": {"file_name": "b.pdf", "section_path": "Results"},
                    },
                ]
            )
        finally:
            settings.qa_source_max_chars = old_source_max
            settings.qa_context_max_chars = old_context_max

        self.assertIn("relevant-A", context)
        self.assertNotIn("representative only", context)
        self.assertIn("fallback", context)
        source_texts = [block.split("]\n", 1)[1] for block in context.split("\n\n")]
        self.assertLessEqual(sum(len(text) for text in source_texts), 20)

    def test_two_stage_query_filters_section_retrieval_to_selected_document(self) -> None:
        store = ChromaVectorStore.__new__(ChromaVectorStore)
        store._last_vector_timings_ms = {}
        store._last_keyword_timings_ms = {}
        store.last_query_timings_ms = {}
        calls: list[list[str] | None] = []

        def fake_vector_query(_query, _client, *, top_k, paper_ids=None, section_types=None):
            calls.append(paper_ids)
            if paper_ids is None:
                return [
                    {"text": "retrieval method", "metadata": {"paper_id": "p1", "parent_id": "s1"}, "distance": 0.1},
                    {"text": "unrelated", "metadata": {"paper_id": "p2", "parent_id": "s2"}, "distance": 0.9},
                ]
            return [
                {
                    "text": "retrieval method details",
                    "metadata": {"paper_id": paper_ids[0], "parent_id": "s1", "chunk_id": "c1"},
                    "distance": 0.1,
                }
            ]

        store.vector_query = fake_vector_query
        old_document_top_k = settings.document_top_k
        settings.document_top_k = 1
        try:
            sections = store.query("retrieval method", object(), top_k=5, retrieval_mode="vector", expand_parent=False)
        finally:
            settings.document_top_k = old_document_top_k

        self.assertEqual(calls, [None, ["p1"]])
        self.assertEqual(len(sections), 1)
        self.assertTrue(sections[0]["aggregated_section"])

    def test_reindex_deletes_stale_chunks_only_after_embedding_succeeds(self) -> None:
        calls: list[tuple[str, object]] = []

        class FakeCollection:
            def delete(self, **kwargs) -> None:
                calls.append(("delete", kwargs))

            def upsert(self, **kwargs) -> None:
                calls.append(("upsert", kwargs))

        class FakeEmbeddingClient:
            def embed_texts(self, texts):
                calls.append(("embed", texts))
                return [[0.1, 0.2] for _ in texts]

        store = ChromaVectorStore.__new__(ChromaVectorStore)
        store.collection = FakeCollection()
        store._keyword_rows = None
        store._keyword_postings = {}
        store._keyword_doc_frequency = {}
        chunk = TextChunk("c1", "paper-1", "paper.pdf", 1, "body", section_path="1 Method")

        indexed = store.index_chunks([chunk], FakeEmbeddingClient(), replace_existing=True)

        self.assertEqual(indexed, 1)
        self.assertEqual([name for name, _ in calls], ["embed", "delete", "upsert"])
        self.assertEqual(calls[1][1], {"where": {"paper_id": "paper-1"}})

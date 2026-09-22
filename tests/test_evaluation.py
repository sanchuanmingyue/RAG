from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from backend.evaluation import (
    BenchmarkCase,
    EvaluationConfig,
    EvaluationRecord,
    _coerce_qrel,
    _citation_validity,
    _citation_gold_alignment,
    _extract_polarity,
    _split_answer_claims,
    build_open_rag_bench_chunks,
    index_chunks_in_batches,
    load_open_rag_bench_cases,
    _parse_judge_output,
    _structured_targets,
    select_open_rag_bench_document_ids,
    run_evaluation,
    score_claim_citations,
    score_answer_correctness,
    summarize_records,
)
from backend.text_splitter import TextChunk
from backend.rag_chain import (
    PaperRAG,
    _diversify_hits_by_paper,
    analyze_question,
    classify_refusal_answer,
    extract_source_citation_ids,
    is_refusal_answer,
    is_vacuous_answer,
)
from backend.prompts import (
    build_compare_messages,
    build_qa_messages,
    is_long_form_question,
    is_multi_paper_question,
    is_yes_no_question,
)


class EvaluationMetricsTests(unittest.TestCase):
    def test_batch_indexing_resumes_and_skips_existing_chunk_ids(self) -> None:
        chunks = [
            TextChunk("c1", "p1", "paper.pdf", 1, "one"),
            TextChunk("c2", "p1", "paper.pdf", 1, "two"),
        ]

        class FakeCollection:
            def get(self, *, ids, include):
                return {"ids": [value for value in ids if value == "c1"]}

        class FakeStore:
            collection = FakeCollection()

            def __init__(self) -> None:
                self.indexed = []

            def index_chunks(self, values, embedding_client):
                self.indexed.extend(chunk.chunk_id for chunk in values)
                return len(values)

        store = FakeStore()
        total = index_chunks_in_batches(store, chunks, object(), batch_size=2)

        self.assertEqual(total, 2)
        self.assertEqual(store.indexed, ["c2"])

    def test_research_background_searches_intro_abstract_and_related_work(self) -> None:
        analysis = analyze_question("请总结多篇论文的研究背景")

        self.assertEqual(analysis.intent, "background")
        self.assertEqual(analysis.section_types, ["abstract", "introduction", "related_work"])

    def test_multi_paper_prompt_groups_sources_by_file(self) -> None:
        hits = [
            {"text": "background A1", "metadata": {"file_name": "a.pdf"}},
            {"text": "background A2", "metadata": {"file_name": "a.pdf"}},
            {"text": "background B", "metadata": {"file_name": "b.pdf"}},
        ]

        prompt = build_qa_messages("总结这些论文的研究背景", hits)[1]["content"]

        self.assertIn("2 篇论文", prompt)
        self.assertIn("a.pdf、b.pdf", prompt)
        self.assertIn("不能把 S1、S2 等来源编号当作不同论文", prompt)

    def test_multi_paper_pronoun_question_is_detected(self) -> None:
        self.assertTrue(is_multi_paper_question("他们分别采用什么方法？"))

    def test_detailed_and_multi_paper_questions_use_long_form(self) -> None:
        self.assertTrue(is_long_form_question("详细解释这篇论文为什么采用双时间尺度设计"))
        self.assertTrue(is_long_form_question("这些论文分别在做什么？"))
        self.assertFalse(is_long_form_question("核心方法是什么？"))

    def test_compare_prompt_requests_mechanism_level_synthesis(self) -> None:
        prompt = build_compare_messages(["paper A", "paper B"])[1]["content"]

        self.assertIn("逐篇解析", prompt)
        self.assertIn("横向对比矩阵", prompt)
        self.assertIn("为什么方法不同", prompt)

    def test_multi_paper_hits_are_balanced_across_documents(self) -> None:
        hits = [
            {"text": "a1", "metadata": {"paper_id": "a"}},
            {"text": "a2", "metadata": {"paper_id": "a"}},
            {"text": "a3", "metadata": {"paper_id": "a"}},
            {"text": "b1", "metadata": {"paper_id": "b"}},
            {"text": "c1", "metadata": {"paper_id": "c"}},
        ]

        balanced = _diversify_hits_by_paper(hits, 4)

        self.assertEqual([hit["text"] for hit in balanced], ["a1", "b1", "c1", "a2"])

    def test_yes_no_prompt_is_only_added_to_binary_questions(self) -> None:
        hits = [{"text": "evidence", "metadata": {}}]
        self.assertTrue(is_yes_no_question("Does the method require labels?"))
        self.assertTrue(is_yes_no_question("该方法是否需要标签？"))
        self.assertFalse(is_yes_no_question("What problem does Reshift address?"))
        self.assertFalse(is_yes_no_question("How does open source code help?"))
        binary_prompt = build_qa_messages("Does it work?", hits)[1]["content"]
        chinese_binary_prompt = build_qa_messages("该方法是否有效？", hits)[1]["content"]
        open_prompt = build_qa_messages("How does it work?", hits)[1]["content"]
        self.assertIn("This is a Yes/No", binary_prompt)
        self.assertIn("Answer strictly in English", binary_prompt)
        self.assertIn("这是一个 Yes/No", chinese_binary_prompt)
        self.assertIn("严格使用中文", chinese_binary_prompt)
        self.assertNotIn("This is a Yes/No", open_prompt)

    def test_open_experiment_prompt_requires_concrete_content(self) -> None:
        hits = [{"text": "Experiments improve the cache hit rate by 41.06%.", "metadata": {}}]

        prompt = build_qa_messages("详细介绍当前文章的实验部分", hits)[1]["content"]

        self.assertIn("开放式内容问题", prompt)
        self.assertIn("实验环境或参数", prompt)
        self.assertIn("不能只说实验有效或存在证据", prompt)

    def test_vacuous_evidence_statement_is_detected(self) -> None:
        self.assertTrue(is_vacuous_answer("是，论文中有明确依据 [S1]。"))
        self.assertTrue(is_vacuous_answer("The sources provide evidence [S1]."))
        self.assertFalse(is_vacuous_answer("缓存命中率至少提升 41.06% [S1]。"))

    def test_chinese_negative_polarity_after_subject_is_detected(self) -> None:
        record = EvaluationRecord(
            case_id="negative-zh", question="Does the method support streaming?",
            gold_doc_id="p", gold_section_id="1", source_type="text", query_type="extractive",
            generation_latency_ms=1.0, answer="该方法不支持流式处理 [S1]。", reference_answer="No.",
        )

        class NoCallClient:
            def embed_texts(self, texts):
                raise AssertionError("polarity scoring must not call embeddings")

        score_answer_correctness(
            [record], NoCallClient(), threshold=0.7, borderline_low=0.65,
            borderline_high=0.75, judge_enabled=True,
        )
        self.assertTrue(record.answer_correct)
        self.assertEqual(record.answer_correctness_stage, "yes_no")

    def test_chinese_negative_polarity_inside_conclusion_is_detected(self) -> None:
        answers = [
            "根据论文片段，该方法通过学习恢复划分。因此，它不需要预先拥有标签信息 [S1]。",
            "根据论文片段，基于冲击矩阵进行标记并不总是直截了当 [S1]。",
            "根据研究结果，经典机器学习方法通常不容易应用于该数据 [S1]。",
        ]
        self.assertEqual([_extract_polarity(answer) for answer in answers], [False, False, False])

    def test_staged_correctness_handles_cross_language_yes_no(self) -> None:
        record = EvaluationRecord(
            case_id="yes",
            question="Does it work?",
            gold_doc_id="p",
            gold_section_id="1",
            source_type="text",
            query_type="extractive",
            generation_latency_ms=1.0,
            answer="是的，该方法有效 [S1]。",
            reference_answer="Yes.",
        )

        class NoCallClient:
            def embed_texts(self, texts):
                raise AssertionError("yes/no scoring must not call embeddings")

        score_answer_correctness(
            [record], NoCallClient(), threshold=0.7, borderline_low=0.65,
            borderline_high=0.75, judge_enabled=True,
        )

        self.assertTrue(record.answer_correct)
        self.assertEqual(record.answer_correctness_stage, "yes_no")

        record.answer = "不需要。"
        record.reference_answer = "No."
        score_answer_correctness(
            [record], NoCallClient(), threshold=0.7, borderline_low=0.65,
            borderline_high=0.75, judge_enabled=True,
        )
        self.assertTrue(record.answer_correct)

    def test_staged_correctness_normalizes_numeric_and_proper_name_answers(self) -> None:
        numeric = EvaluationRecord(
            case_id="number", question="What is the value?", gold_doc_id="p", gold_section_id="1",
            source_type="text", query_type="extractive", generation_latency_ms=1.0,
            answer="该数值为 42 [S1]。", reference_answer="The value is 42.",
        )
        proper = EvaluationRecord(
            case_id="name", question="Which algorithm is used?", gold_doc_id="p", gold_section_id="1",
            source_type="text", query_type="extractive", generation_latency_ms=1.0,
            answer="系统使用 BFGS 算法 [S1]。", reference_answer="The algorithm is BFGS.",
        )

        class NoCallClient:
            def embed_texts(self, texts):
                raise AssertionError("structured scoring must not call embeddings")

        score_answer_correctness(
            [numeric, proper], NoCallClient(), threshold=0.7, borderline_low=0.65,
            borderline_high=0.75, judge_enabled=True,
        )

        self.assertTrue(numeric.answer_correct)
        self.assertTrue(proper.answer_correct)
        self.assertEqual(numeric.answer_correctness_stage, "structured_exact")
        self.assertEqual(proper.answer_correctness_stage, "structured_exact")

    def test_proper_name_ignores_leading_article_and_mismatch_falls_back_to_semantics(self) -> None:
        self.assertEqual(
            _structured_targets(
                "What problem does the Reshift operation aim to address?",
                "The Reshift operation minimizes intrinsic ambiguities.",
            ),
            (None, []),
        )
        article = EvaluationRecord(
            case_id="article", question="Which operation is used?", gold_doc_id="p", gold_section_id="1",
            source_type="text", query_type="extractive", generation_latency_ms=1.0,
            answer="Reshift is used [S1].", reference_answer="The Reshift operation is used.",
        )
        fallback = EvaluationRecord(
            case_id="fallback", question="Which algorithm is used?", gold_doc_id="p", gold_section_id="1",
            source_type="text", query_type="extractive", generation_latency_ms=1.0,
            answer="It uses a quasi-Newton method [S1].", reference_answer="The algorithm is BFGS.",
        )

        class SemanticClient:
            def embed_texts(self, texts):
                return [[1.0, 0.0] for _ in texts]

        score_answer_correctness(
            [article, fallback], SemanticClient(), threshold=0.7, borderline_low=0.65,
            borderline_high=0.75, judge_enabled=False,
        )
        self.assertTrue(article.answer_correct)
        self.assertEqual(article.answer_correctness_stage, "structured_exact")
        self.assertTrue(fallback.answer_correct)
        self.assertEqual(fallback.answer_correctness_stage, "semantic_similarity")

    def test_numeric_boundaries_allow_chinese_adjacency_but_not_partial_matches(self) -> None:
        matching = EvaluationRecord(
            case_id="zh-number", question="数值是多少？", gold_doc_id="p", gold_section_id="1",
            source_type="text", query_type="extractive", generation_latency_ms=1.0,
            answer="该数值为42个 [S1]。", reference_answer="The value is 42.",
        )
        wrong = EvaluationRecord(
            case_id="zh-number-wrong", question="数值是多少？", gold_doc_id="p", gold_section_id="1",
            source_type="text", query_type="extractive", generation_latency_ms=1.0,
            answer="该数值为420个 [S1]。", reference_answer="The value is 42.",
        )

        class NoCallClient:
            def embed_texts(self, texts):
                raise AssertionError("numeric scoring must not call embeddings")

        score_answer_correctness(
            [matching, wrong], NoCallClient(), threshold=0.7, borderline_low=0.65,
            borderline_high=0.75, judge_enabled=True,
        )
        self.assertTrue(matching.answer_correct)
        self.assertFalse(wrong.answer_correct)

    def test_judge_json_parser_tolerates_fences_single_quotes_and_trailing_commas(self) -> None:
        correct, reason = _parse_judge_output(
            "评审结果如下：\n```json\n{'correct': 'yes', 'reason': '语义一致',}\n```"
        )
        self.assertTrue(correct)
        self.assertEqual(reason, "语义一致")

        correct, reason = _parse_judge_output('Result: {"is_correct": false, "explanation": "wrong number",}')
        self.assertFalse(correct)
        self.assertEqual(reason, "wrong number")

    def test_borderline_semantic_similarity_uses_llm_judge(self) -> None:
        record = EvaluationRecord(
            case_id="judge", question="Why is it useful?", gold_doc_id="p", gold_section_id="1",
            source_type="text", query_type="abstractive", generation_latency_ms=1.0,
            answer="它提高了系统稳定性 [S1]。", reference_answer="It improves system stability.",
        )

        class JudgeClient:
            last_chat_metadata = {"model_used": "judge-model"}

            def embed_texts(self, texts):
                return [[1.0, 0.0], [0.70, 0.714142842]]

            def chat(self, messages, temperature=0.0, *, model_candidates=None):
                return '{"correct": true, "reason": "Same meaning across languages."}'

        score_answer_correctness(
            [record], JudgeClient(), threshold=0.7, borderline_low=0.65,
            borderline_high=0.75, judge_enabled=True, judge_models=["judge-model"],
        )

        self.assertTrue(record.answer_correct)
        self.assertEqual(record.answer_correctness_stage, "llm_judge")
        self.assertEqual(record.judge_model_used, "judge-model")
        self.assertEqual(record.judge_reason, "Same meaning across languages.")

    def test_semantic_refusal_detection_is_not_exact_string_only(self) -> None:
        self.assertTrue(is_refusal_answer("提供的论文片段未包含足够信息，因此无法回答。"))
        self.assertTrue(is_refusal_answer("There is not enough information in the provided context to answer."))
        self.assertFalse(is_refusal_answer("The method cannot use MLE because the normalizer is intractable [S1]."))

    def test_mixed_refusal_is_distinguished_from_pure_refusal(self) -> None:
        mixed = "Reshift resolves phase ambiguities before crossover [S1].\n\n论文中没有找到明确依据。"
        self.assertEqual(classify_refusal_answer(mixed), "mixed_refusal")
        self.assertFalse(is_refusal_answer(mixed))
        self.assertEqual(classify_refusal_answer("论文中没有找到明确依据。"), "pure_refusal")
        self.assertTrue(is_refusal_answer("论文中没有找到明确依据。"))
        explained_refusal = (
            "论文片段中未提供关于训练目标的明确依据。\n"
            "[S1] 提到了该方法，但未说明其具体最小化目标。\n"
            "[S2] 与问题中的训练目标无关。\n"
            "因此，根据给定片段无法回答该问题。"
        )
        self.assertEqual(classify_refusal_answer(explained_refusal), "pure_refusal")

    def test_generation_records_model_and_missing_citation_refusals_separately(self) -> None:
        class FakeClient:
            def __init__(self, answer: str) -> None:
                self.answer = answer

            def chat(self, messages, temperature=0.2):
                return self.answer

        hits = [{"text": "Direct evidence", "metadata": {"paper_id": "p"}}]
        model_refusal = PaperRAG(object(), FakeClient("论文片段中未提供回答所需的信息。"))  # type: ignore[arg-type]
        spaced_citation = PaperRAG(object(), FakeClient("The supported answer [ S 1 ]."))  # type: ignore[arg-type]
        missing_citation = PaperRAG(object(), FakeClient("The supported answer."))  # type: ignore[arg-type]
        mixed_refusal = PaperRAG(  # type: ignore[arg-type]
            object(), FakeClient("The supported answer contains evidence [S1].\n\n论文中没有找到明确依据。")
        )
        invalid_citation = PaperRAG(object(), FakeClient("The supported answer [S2, S1]."))  # type: ignore[arg-type]
        invalid_only = PaperRAG(object(), FakeClient("The unsupported answer [S2]."))  # type: ignore[arg-type]

        refused = model_refusal.answer_from_hits("question", hits)
        accepted = spaced_citation.answer_from_hits("question", hits)
        forced = missing_citation.answer_from_hits("question", hits)
        mixed = mixed_refusal.answer_from_hits("question", hits)
        invalid = invalid_citation.answer_from_hits("question", hits)
        invalid_forced = invalid_only.answer_from_hits("question", hits)

        self.assertEqual(refused["refusal_reason"], "model_refusal")
        self.assertEqual(refused["refusal_type"], "pure_refusal")
        self.assertIsNone(accepted["refusal_reason"])
        self.assertEqual(forced["refusal_reason"], "missing_citation")
        self.assertEqual(forced["answer"], "I could not find explicit evidence in the paper.")
        self.assertEqual(forced["raw_answer"], "The supported answer.")
        self.assertIsNone(mixed["refusal_reason"])
        self.assertEqual(mixed["refusal_type"], "mixed_refusal")
        self.assertEqual(mixed["answer"], "The supported answer contains evidence [S1].")
        self.assertIn("论文中没有找到明确依据", mixed["raw_answer"])
        self.assertIsNone(invalid["refusal_reason"])
        self.assertEqual(invalid["invalid_citation_ids"], [2])
        self.assertEqual(invalid["answer"], "The supported answer [S1].")
        self.assertTrue(invalid["citation_repaired"])
        self.assertEqual(invalid_forced["refusal_reason"], "invalid_citation")
        self.assertTrue(is_refusal_answer("I could not find explicit evidence in the paper."))

    def test_missing_citation_gets_one_corrective_retry(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages, temperature=0.2):
                self.calls += 1
                if self.calls == 1:
                    return "The method uses graph reinforcement learning."
                return "The method uses graph reinforcement learning [S1]."

        client = FakeClient()
        result = PaperRAG(object(), client).answer_from_hits(  # type: ignore[arg-type]
            "What method is used?",
            [{"text": "The method uses graph reinforcement learning.", "metadata": {}}],
        )

        self.assertEqual(client.calls, 2)
        self.assertTrue(result["citation_retry_attempted"])
        self.assertIsNone(result["refusal_reason"])
        self.assertEqual(result["answer"], "The method uses graph reinforcement learning [S1].")

    def test_vacuous_answer_gets_content_retry(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages, temperature=0.2):
                self.calls += 1
                if self.calls == 1:
                    return "是，论文中有明确依据 [S1]。"
                return "实验表明，该方法将缓存命中率至少提升了 41.06% [S1]。"

        client = FakeClient()
        result = PaperRAG(object(), client).answer_from_hits(  # type: ignore[arg-type]
            "这篇论文的主要实验结论是什么？",
            [{"text": "Experimental results improve the cache hit rate by at least 41.06%.", "metadata": {}}],
        )

        self.assertEqual(client.calls, 2)
        self.assertTrue(result["answer_quality_retry_attempted"])
        self.assertEqual(result["answer_quality_retry_reason"], "vacuous_answer")
        self.assertIn("41.06%", result["answer"])
        self.assertFalse(result["answer"].startswith("是，"))
        self.assertIsNone(result["refusal_reason"])

    def test_premature_experiment_refusal_gets_evidence_retry(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages, temperature=0.2):
                self.calls += 1
                if self.calls == 1:
                    return "论文中没有找到明确依据。"
                return "实验在服务器环境中进行，并比较了缓存命中率和服务成本 [S1]。"

        client = FakeClient()
        result = PaperRAG(object(), client).answer_from_hits(  # type: ignore[arg-type]
            "详细介绍当前文章的实验部分",
            [
                {
                    "text": "Experimental evaluation on a server compares the cache hit rate and service cost.",
                    "metadata": {"section_type": "method"},
                }
            ],
        )

        self.assertEqual(client.calls, 2)
        self.assertTrue(result["answer_quality_retry_attempted"])
        self.assertEqual(result["answer_quality_retry_reason"], "premature_refusal")
        self.assertIsNone(result["refusal_reason"])

    def test_premature_background_refusal_retries_when_intro_was_retrieved(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages, temperature=0.2):
                self.calls += 1
                if self.calls == 1:
                    return "论文中没有找到明确依据。"
                return "大模型推理资源需求高，边缘负载与云端扩缩容相互影响 [S1]。"

        client = FakeClient()
        result = PaperRAG(object(), client).answer_from_hits(  # type: ignore[arg-type]
            "介绍文章的研究背景",
            [
                {
                    "text": "LLM services require more resources, and inference latency grows nonlinearly.",
                    "metadata": {"section_type": "introduction"},
                }
            ],
        )

        self.assertEqual(client.calls, 2)
        self.assertTrue(result["answer_quality_retry_attempted"])
        self.assertEqual(result["answer_quality_retry_reason"], "premature_refusal")
        self.assertIsNone(result["refusal_reason"])

    def test_long_form_qa_builds_private_evidence_plan_and_uses_large_budget(self) -> None:
        class FakeConfig:
            qa_planning_enabled = True
            qa_plan_max_tokens = 77
            qa_long_max_tokens = 333
            qa_max_tokens = 50
            qa_auto_continue = False

        class FakeClient:
            config = FakeConfig()
            last_chat_metadata = {"finish_reason": "stop"}

            def __init__(self) -> None:
                self.calls = []

            def chat(self, messages, temperature=0.2, max_tokens=None):
                self.calls.append({"messages": messages, "max_tokens": max_tokens})
                if len(self.calls) == 1:
                    return "方法动机与机制 [S1]"
                return "该方法通过双时间尺度分离缓存与资源分配，以匹配两类状态的变化速度 [S1]。"

        client = FakeClient()
        result = PaperRAG(object(), client).answer_from_hits(  # type: ignore[arg-type]
            "详细解释这篇论文为什么采用双时间尺度设计",
            [{"text": "Caching changes slowly while channels change quickly.", "metadata": {}}],
        )

        self.assertEqual([call["max_tokens"] for call in client.calls], [77, 333])
        self.assertTrue(result["long_form"])
        self.assertTrue(result["evidence_plan_used"])
        self.assertEqual(result["max_output_tokens"], 333)

    def test_long_form_qa_continues_once_after_length_finish(self) -> None:
        class FakeConfig:
            qa_planning_enabled = False
            qa_long_max_tokens = 200
            qa_max_tokens = 50
            qa_auto_continue = True

        class FakeClient:
            config = FakeConfig()

            def __init__(self) -> None:
                self.calls = 0
                self.last_chat_metadata = {}

            def chat(self, messages, temperature=0.2, max_tokens=None):
                self.calls += 1
                if self.calls == 1:
                    self.last_chat_metadata = {"finish_reason": "length"}
                    return "第一部分介绍系统模型 [S1]。"
                self.last_chat_metadata = {"finish_reason": "stop"}
                return "第二部分补充实验验证 [S1]。"

        client = FakeClient()
        result = PaperRAG(object(), client).answer_from_hits(  # type: ignore[arg-type]
            "详细介绍全文",
            [{"text": "System model and experiments.", "metadata": {}}],
        )

        self.assertEqual(client.calls, 2)
        self.assertTrue(result["continuation_attempted"])
        self.assertIn("第一部分", result["answer"])
        self.assertIn("第二部分", result["answer"])

    def test_detailed_question_retrieves_multiple_evidence_facets(self) -> None:
        class FakeStore:
            def __init__(self) -> None:
                self.queries = []
                self.last_query_timings_ms = {}

            def query(self, **kwargs):
                query = kwargs["query_text"]
                self.queries.append(query)
                index = len(self.queries)
                self.last_query_timings_ms = {"total_ms": 1.0}
                return [
                    {
                        "text": f"evidence {index}",
                        "metadata": {
                            "paper_id": "p1",
                            "parent_id": f"section-{index}",
                            "section_type": "method",
                        },
                    }
                ]

        store = FakeStore()
        hits, _analysis = PaperRAG(store, object()).retrieve_hits(  # type: ignore[arg-type]
            "请逐章详细介绍全文的背景、方法和实验",
            paper_id="p1",
            top_k=12,
        )

        self.assertGreaterEqual(len(store.queries), 5)
        self.assertGreaterEqual(len(hits), 5)
        self.assertIn("逐章详细介绍", store.queries[-1])

    def test_suspicious_binary_comparison_gets_one_consistency_retry(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages, temperature=0.2):
                self.calls += 1
                if self.calls == 1:
                    return "否，相反，Hamming distance <= prefix distance [S1]."
                return "是，Hamming distance <= prefix distance，因此 prefix distance >= Hamming distance [S1]."

        client = FakeClient()
        rag = PaperRAG(object(), client)  # type: ignore[arg-type]
        result = rag.answer_from_hits(
            "Is prefix distance greater than or equal to Hamming distance?",
            [{"text": "Hamming distance <= prefix distance", "metadata": {}}],
        )
        self.assertEqual(client.calls, 2)
        self.assertTrue(result["binary_consistency_retried"])
        self.assertTrue(result["answer"].startswith("是"))

    def test_claim_level_citation_metrics_count_unsupported_claims(self) -> None:
        record = EvaluationRecord(
            case_id="claims", question="q", gold_doc_id="p", gold_section_id="1",
            source_type="text", query_type="extractive", generation_latency_ms=1.0,
            answer="The method improves accuracy [S1]. An unrelated unsupported assertion.",
        )

        class SameVectorClient:
            def embed_texts(self, texts):
                return [[1.0, 0.0] for _ in texts]

        score_claim_citations(
            [record],
            {"claims": [{"text": "The method improves accuracy."}]},
            SameVectorClient(),  # type: ignore[arg-type]
            similarity_threshold=0.55,
        )
        self.assertEqual(record.claim_count, 2)
        self.assertEqual(record.cited_claim_count, 1)
        self.assertEqual(record.claim_citation_precision, 1.0)
        self.assertEqual(record.claim_citation_recall, 0.5)
        self.assertEqual(record.unsupported_claim_rate, 0.5)

    def test_claim_split_ignores_colon_lead_ins_and_inherits_paragraph_citation(self) -> None:
        claims = _split_answer_claims(
            "主要结论：\n"
            "Accuracy improves substantially. Recall also rises [S1].\n"
            "A separate paragraph remains unsupported."
        )

        self.assertEqual(len(claims), 3)
        self.assertEqual(extract_source_citation_ids(claims[0]), [1])
        self.assertEqual(extract_source_citation_ids(claims[1]), [1])
        self.assertEqual(extract_source_citation_ids(claims[2]), [])

    def test_claim_level_citation_metrics_deduplicate_source_ids(self) -> None:
        record = EvaluationRecord(
            case_id="duplicate-citations", question="q", gold_doc_id="p", gold_section_id="1",
            source_type="text", query_type="extractive", generation_latency_ms=1.0,
            answer="The method improves accuracy [S1, S1].",
        )

        class SameVectorClient:
            def embed_texts(self, texts):
                return [[1.0, 0.0] for _ in texts]

        score_claim_citations(
            [record],
            {"duplicate-citations": [{"text": "The method improves accuracy."}]},
            SameVectorClient(),  # type: ignore[arg-type]
            similarity_threshold=0.55,
        )
        self.assertEqual(record.claim_citation_details[0]["citation_ids"], [1])
        self.assertEqual(record.citation_link_count, 1)

    def test_generation_evaluation_reuses_retrieval_and_scores_answer(self) -> None:
        class FakeStore:
            def __init__(self) -> None:
                self.query_count = 0
                self.last_query_timings_ms = {"total_ms": 1.0}

            def query(self, **kwargs):
                self.query_count += 1
                return [
                    {
                        "text": "The method uses retrieval.",
                        "metadata": {
                            "paper_id": "paper-1",
                            "benchmark_doc_id": "paper-1",
                            "benchmark_section_id": "0",
                        },
                    }
                ]

        class FakeClient:
            last_chat_metadata = {
                "model_used": "fake-model",
                "attempted_models": ["fake-model"],
                "fallback_count": 0,
            }

            def chat(self, messages):
                return "The method uses retrieval [S1]."

            def embed_texts(self, texts):
                return [[1.0, 0.0] for _ in texts]

        case = BenchmarkCase(
            case_id="case-1",
            question="What method is used?",
            gold_doc_id="paper-1",
            gold_section_id="0",
            reference_answer="The method uses retrieval.",
        )
        store = FakeStore()
        records = run_evaluation(
            [case],
            store,  # type: ignore[arg-type]
            FakeClient(),  # type: ignore[arg-type]
            EvaluationConfig(run_generation=True, answer_correctness_threshold=0.7),
        )

        self.assertEqual(store.query_count, 1)
        self.assertTrue(records[0].answer_correct)
        self.assertEqual(records[0].answer_similarity, 1.0)
        self.assertEqual(records[0].refusal_reason, None)

    def test_summary_calculates_doc_and_section_metrics(self) -> None:
        records = [
            EvaluationRecord(
                case_id="a",
                question="q1",
                gold_doc_id="doc-a",
                gold_section_id="0",
                source_type="text",
                query_type="extractive",
                doc_rank=1,
                section_rank=2,
                retrieval_latency_ms=10,
            ),
            EvaluationRecord(
                case_id="b",
                question="q2",
                gold_doc_id="doc-b",
                gold_section_id="1",
                source_type="text",
                query_type="extractive",
                doc_rank=None,
                section_rank=None,
                retrieval_latency_ms=30,
            ),
        ]

        summary = summarize_records(records, top_k=5)

        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["doc_recall_at_k"], 0.5)
        self.assertEqual(summary["section_recall_at_k"], 0.5)
        self.assertEqual(summary["doc_mrr"], 0.5)
        self.assertEqual(summary["section_mrr"], 0.25)
        self.assertEqual(summary["retrieval_latency_ms"]["p50"], 10)

    def test_citation_validation_checks_source_range(self) -> None:
        self.assertTrue(_citation_validity("Evidence [S1] and [S2]", 2))
        self.assertTrue(_citation_validity("Evidence [ S1 ] and [S 2]", 2))
        self.assertTrue(_citation_validity("Evidence [ s 1 ]", 1))
        self.assertTrue(_citation_validity("Grouped [S1, S2; S3]", 3))
        self.assertTrue(_citation_validity("Range [S1-S3]", 3))
        self.assertEqual(extract_source_citation_ids("Mixed [S1, S3-S5; S2]"), [1, 3, 4, 5, 2])
        self.assertFalse(_citation_validity("Invalid [S3]", 2))
        self.assertFalse(_citation_validity("Invalid grouped [S6, S5]", 5))
        self.assertFalse(_citation_validity("Invalid range [S1-S3]", 2))
        self.assertIsNone(_citation_validity("No citations", 2))

    def test_citation_alignment_checks_cited_gold_section(self) -> None:
        sources = [
            {"metadata": {"benchmark_doc_id": "other", "benchmark_section_id": "1"}},
            {"metadata": {"benchmark_doc_id": "gold", "benchmark_section_id": "3"}},
        ]
        self.assertEqual(_citation_gold_alignment("Evidence [S2]", sources, "gold", "3"), (True, True))
        self.assertEqual(_citation_gold_alignment("Evidence [S1, S2]", sources, "gold", "3"), (True, True))
        self.assertEqual(_citation_gold_alignment("Evidence [S1]", sources, "gold", "3"), (False, False))
        self.assertEqual(_citation_gold_alignment("Bad [S3]", sources, "gold", "3"), (None, None))

    def test_qrel_preserves_zero_section_id(self) -> None:
        self.assertEqual(_coerce_qrel({"doc_id": "paper-1", "section_id": 0}), ("paper-1", "0"))

    def test_open_rag_bench_adapter_keeps_gold_section_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "corpus").mkdir()
            (root / "queries.json").write_text(
                json.dumps(
                    {
                        "text-case": {"query": "What is the method?", "source": "text", "type": "extractive"},
                        "image-case": {"query": "What does the figure show?", "source": "text-image"},
                    }
                ),
                encoding="utf-8",
            )
            (root / "qrels.json").write_text(
                json.dumps({"text-case": {"doc_id": "paper-1", "section_id": 0}, "image-case": {"doc_id": "paper-1", "section_id": 0}}),
                encoding="utf-8",
            )
            (root / "answers.json").write_text(json.dumps({"text-case": "A method"}), encoding="utf-8")
            (root / "corpus" / "paper-1.json").write_text(
                json.dumps(
                    {
                        "id": "paper-1",
                        "title": "Test paper",
                        "sections": [{"section_id": 0, "text": "## 2. Method\nThe method uses retrieval."}],
                    }
                ),
                encoding="utf-8",
            )

            cases = load_open_rag_bench_cases(root, text_only=True)
            chunks = build_open_rag_bench_chunks(root, chunk_size=100, chunk_overlap=10)
            document_ids = select_open_rag_bench_document_ids(root, cases, max_documents=1)

        self.assertEqual([case.case_id for case in cases], ["text-case"])
        self.assertEqual(chunks[0].to_metadata()["benchmark_doc_id"], "paper-1")
        self.assertEqual(chunks[0].to_metadata()["benchmark_section_id"], "0")
        self.assertEqual(chunks[0].to_metadata()["section_title"], "2. Method")
        self.assertEqual(chunks[0].to_metadata()["section_path"], "2. Method")
        self.assertEqual(chunks[0].to_metadata()["section_type"], "method")
        self.assertEqual(document_ids, {"paper-1"})

    def test_open_rag_bench_adapter_extracts_same_line_heading_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "corpus").mkdir()
            for name in ("queries.json", "qrels.json", "answers.json"):
                (root / name).write_text("{}", encoding="utf-8")
            (root / "corpus" / "paper.json").write_text(
                json.dumps(
                    {
                        "id": "paper",
                        "sections": [
                            {"text": "## III. Experiments ## A. Ablation Study\nResults are reported here."}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            chunks = build_open_rag_bench_chunks(root, chunk_size=200, chunk_overlap=10)

        metadata = chunks[0].to_metadata()
        self.assertEqual(metadata["section_title"], "A. Ablation Study")
        self.assertEqual(metadata["section_path"], "III. Experiments > A. Ablation Study")
        self.assertEqual(metadata["section_type"], "result")


if __name__ == "__main__":
    unittest.main()

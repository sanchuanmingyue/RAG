"""Run the Open RAG Benchmark retrieval evaluation against PaperReader-RAG.

The default mode only evaluates retrieval.  Pass --run-generation only after a
small retrieval run looks healthy, because it calls the configured chat model
once per benchmark question.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


# 允许从项目根目录直接执行 ``python scripts/run_open_rag_bench.py``。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.embeddings import OpenAICompatibleClient
from backend.evaluation import (
    EvaluationConfig,
    build_open_rag_bench_chunks,
    index_chunks_in_batches,
    load_open_rag_bench_cases,
    run_evaluation,
    select_open_rag_bench_document_ids,
    summarize_records,
    write_evaluation_report,
)
from backend.vector_store import ChromaVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PaperReader-RAG with Open RAG Benchmark.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="下载后的 Open RAG Benchmark 根目录。")
    parser.add_argument("--collection", default="open_ragbench_eval", help="专用 Chroma collection 名称。")
    parser.add_argument("--run-name", default="open_ragbench", help="结果文件名前缀。")
    parser.add_argument("--max-cases", type=int, default=100, help="确定性抽样的题数。")
    parser.add_argument(
        "--max-documents",
        type=int,
        default=200,
        help="评测语料文档数；自动保留所有标准论文，0 表示完整 benchmark。",
    )
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子。")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-mode", choices=("vector", "keyword", "hybrid"), default="hybrid")
    parser.add_argument("--include-multimodal", action="store_true", help="包含表格/图片相关问题；主文本 RAG 首轮不建议启用。")
    parser.add_argument("--include-tables", action="store_true", help="把结构化 corpus 中的 Markdown 表格拼入待索引文本。")
    parser.add_argument("--rebuild", action="store_true", help="清空此专用 collection 后重建索引。")
    parser.add_argument("--run-generation", action="store_true", help="额外运行回答生成，产生 API 成本。")
    parser.add_argument(
        "--generation-max-cases",
        type=int,
        default=0,
        help="仅对确定语料后的前 N 题执行评测；0 表示全部。适合低额度生成/引用测试。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k 必须大于 0。")
    if args.max_cases <= 0:
        raise SystemExit("--max-cases 必须大于 0。")
    if args.max_documents < 0:
        raise SystemExit("--max-documents 不能小于 0。")
    if args.generation_max_cases < 0:
        raise SystemExit("--generation-max-cases 不能小于 0。")
    if args.generation_max_cases and not args.run_generation:
        raise SystemExit("--generation-max-cases 需要与 --run-generation 一起使用。")
    if not 0 <= settings.answer_correctness_threshold <= 1:
        raise SystemExit("ANSWER_CORRECTNESS_THRESHOLD 必须在 0 到 1 之间。")
    if not (
        0 <= settings.answer_correctness_borderline_low
        <= settings.answer_correctness_threshold
        <= settings.answer_correctness_borderline_high
        <= 1
    ):
        raise SystemExit(
            "答案正确性阈值必须满足 0 <= BORDERLINE_LOW <= THRESHOLD <= BORDERLINE_HIGH <= 1。"
        )
    if not 0 <= settings.claim_citation_similarity_threshold <= 1:
        raise SystemExit("CLAIM_CITATION_SIMILARITY_THRESHOLD 必须在 0 到 1 之间。")
    if not settings.is_ready:
        raise SystemExit(
            "请分别配置 LLM_API_KEY/LLM_BASE_URL/LLM_MODEL 和 "
            "EMBEDDING_API_KEY/EMBEDDING_BASE_URL/EMBEDDING_MODEL。"
        )

    cases = load_open_rag_bench_cases(
        args.dataset_root,
        max_cases=args.max_cases,
        seed=args.seed,
        text_only=not args.include_multimodal,
    )
    if not cases:
        raise SystemExit("筛选后没有可评测的问题。")
    print(f"已加载 {len(cases)} 道评测题（seed={args.seed}）。", flush=True)
    document_ids = select_open_rag_bench_document_ids(
        args.dataset_root,
        cases,
        max_documents=args.max_documents,
        seed=args.seed,
    )
    print(f"本次评测将索引 {len(document_ids)} 篇文档。", flush=True)
    evaluation_cases = cases
    if args.run_generation and args.generation_max_cases:
        evaluation_cases = cases[: args.generation_max_cases]
        print(
            f"低额度生成模式：语料仍按 {len(cases)} 题确定，仅生成并评测前 {len(evaluation_cases)} 题。",
            flush=True,
        )

    vector_store = ChromaVectorStore(collection_name=args.collection)
    client = OpenAICompatibleClient()
    if args.rebuild:
        vector_store.reset_collection()
    print("正在读取 benchmark corpus 并切分文本...", flush=True)
    chunks = build_open_rag_bench_chunks(
        args.dataset_root,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        include_tables=args.include_tables,
        document_ids=document_ids,
    )
    expected_ids = {chunk.chunk_id for chunk in chunks}
    expected_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    existing_payload = vector_store.collection.get(include=["documents", "metadatas"])
    existing_ids = {str(value) for value in existing_payload.get("ids", [])}
    unexpected_ids = existing_ids - expected_ids
    existing_rows = {
        str(row_id): (str(document or ""), dict(metadata or {}))
        for row_id, document, metadata in zip(
            existing_payload.get("ids", []),
            existing_payload.get("documents") or [],
            existing_payload.get("metadatas") or [],
        )
    }
    mismatched_ids = {
        row_id
        for row_id, (document, metadata) in existing_rows.items()
        if row_id in expected_by_id
        and (
            document != expected_by_id[row_id].text
            or metadata.get("section_path", "") != expected_by_id[row_id].section_path
            or metadata.get("file_name", "") != expected_by_id[row_id].file_name
        )
    }
    if unexpected_ids or mismatched_ids:
        raise SystemExit(
            f"collection={args.collection} 与当前段落感知切分不一致："
            f"多余 {len(unexpected_ids)} 个、内容不匹配 {len(mismatched_ids)} 个 chunk。"
            "请更换 --collection，或添加 --rebuild 后重试。"
        )
    missing_count = len(expected_ids - existing_ids)
    if missing_count:
        print(f"正在生成 embedding 并写入 Chroma（共 {len(chunks)} chunks）...", flush=True)
        if existing_ids:
            print(
                f"检测到上次已完成 {len(existing_ids)} chunks，本次从剩余 {missing_count} chunks 继续。",
                flush=True,
            )

        def show_index_progress(indexed: int, total: int) -> None:
            print(f"  索引进度：{indexed}/{total} chunks", flush=True)

        indexed = index_chunks_in_batches(
            vector_store,
            chunks,
            client,
            resume=True,
            progress_callback=show_index_progress,
        )
        print(
            f"已建立 benchmark 索引：{indexed} chunks，{len(document_ids)} 篇文档，collection={args.collection}",
            flush=True,
        )
    else:
        print(f"复用完整 benchmark 索引：{len(existing_ids)} chunks，collection={args.collection}", flush=True)

    if settings.enable_startup_warmup:
        print("正在预热关键词索引和章节重排器...", flush=True)
        warmup = vector_store.warmup()
        print(json.dumps(warmup, ensure_ascii=False), flush=True)

    config = EvaluationConfig(
        top_k=args.top_k,
        retrieval_mode=args.retrieval_mode,
        expand_parent=False,
        run_generation=args.run_generation,
        corpus_document_count=len(document_ids),
        embedding_model=settings.embedding_model,
        collection_name=args.collection,
        chunk_strategy="section_paragraph",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        two_stage_retrieval=settings.enable_two_stage_retrieval,
        document_candidate_k=settings.document_candidate_k,
        document_top_k=settings.document_top_k,
        section_candidate_k=settings.section_candidate_k,
        section_support_chunks=settings.section_support_chunks,
        reference_section_penalty=settings.reference_section_penalty,
        section_index=settings.enable_section_index,
        keyword_query_cache_size=settings.keyword_query_cache_size,
        cross_encoder_reranker=settings.enable_cross_encoder_reranker,
        cross_encoder_model=settings.cross_encoder_model,
        cross_encoder_candidate_k=settings.cross_encoder_candidate_k,
        cross_encoder_weight=settings.cross_encoder_weight,
        section_reranker_provider=settings.reranker_provider,
        section_reranker_model=settings.reranker_model if settings.reranker_provider != "local" else settings.cross_encoder_model,
        section_reranker_candidate_k=settings.reranker_candidate_k,
        section_reranker_weight=settings.reranker_weight,
        answer_correctness_threshold=settings.answer_correctness_threshold,
        answer_correctness_borderline_low=settings.answer_correctness_borderline_low,
        answer_correctness_borderline_high=settings.answer_correctness_borderline_high,
        answer_judge_enabled=settings.answer_judge_enabled,
        answer_judge_models=tuple(settings.answer_judge_models),
        claim_citation_eval_enabled=settings.claim_citation_eval_enabled,
        claim_citation_similarity_threshold=settings.claim_citation_similarity_threshold,
    )
    print("正在执行检索评测...", flush=True)

    def show_evaluation_progress(completed: int, total: int) -> None:
        if completed == total or completed % 10 == 0:
            print(f"  评测进度：{completed}/{total} 题", flush=True)

    records = run_evaluation(evaluation_cases, vector_store, client, config, progress_callback=show_evaluation_progress)
    report_path = write_evaluation_report(records, config, run_name=args.run_name)
    print(
        json.dumps(
            summarize_records(
                records,
                args.top_k,
                answer_correctness_threshold=config.answer_correctness_threshold,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    print(f"逐题结果：{report_path}", flush=True)


if __name__ == "__main__":
    main()

"""Run compact QASPER and ScholarQA-Multi acceptance evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.embeddings import OpenAICompatibleClient
from backend.evaluation import index_chunks_in_batches
from backend.literature_evaluation import (
    build_qasper_chunks,
    load_qasper_cases,
    load_scholarqa_multi_cases,
    run_literature_evaluation,
    write_literature_report,
)
from backend.vector_store import ChromaVectorStore


DEFAULT_QASPER = PROJECT_ROOT / "data" / "QASPER"
DEFAULT_SCHOLARQA_MULTI = (
    PROJECT_ROOT / "data" / "ScholarQABench" / "data" / "scholarqa_multi" / "human_answers.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate practical scientific-paper reading abilities.")
    parser.add_argument(
        "--suite",
        required=True,
        choices=("qasper", "qasper-unanswerable", "scholarqa-multi"),
    )
    parser.add_argument("--dataset", type=Path, help="QASPER split JSON or ScholarQA-Multi JSON.")
    parser.add_argument("--max-cases", type=int, help="Defaults: QASPER 30, unanswerable 15, ScholarQA-Multi 10.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, help="Defaults: QASPER 5, ScholarQA-Multi 10.")
    parser.add_argument("--retrieval-mode", choices=("vector", "keyword", "hybrid"), default="hybrid")
    parser.add_argument("--collection", help="Dedicated Chroma collection; a suite-specific default is used.")
    parser.add_argument("--run-name", help="Output report prefix.")
    parser.add_argument("--correctness-threshold", type=float)
    parser.add_argument("--borderline-low", type=float)
    parser.add_argument("--borderline-high", type=float)
    parser.add_argument("--rebuild", action="store_true", help="Reset and rebuild this dedicated collection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    top_k = args.top_k if args.top_k is not None else (10 if args.suite == "scholarqa-multi" else 5)
    if top_k <= 0:
        raise SystemExit("--top-k 必须大于 0。")
    defaults = {"qasper": 30, "qasper-unanswerable": 15, "scholarqa-multi": 10}
    max_cases = args.max_cases if args.max_cases is not None else defaults[args.suite]
    if max_cases <= 0:
        raise SystemExit("--max-cases 必须大于 0。")
    if not settings.is_ready:
        raise SystemExit("请先配置生成模型和 embedding 模型 API。")
    qasper_thresholds = (0.65, 0.40, 0.78)
    default_thresholds = (
        settings.answer_correctness_threshold,
        settings.answer_correctness_borderline_low,
        settings.answer_correctness_borderline_high,
    )
    threshold, borderline_low, borderline_high = (
        qasper_thresholds if args.suite == "qasper" else default_thresholds
    )
    threshold = args.correctness_threshold if args.correctness_threshold is not None else threshold
    borderline_low = args.borderline_low if args.borderline_low is not None else borderline_low
    borderline_high = args.borderline_high if args.borderline_high is not None else borderline_high
    if not 0 <= borderline_low <= threshold <= borderline_high <= 1:
        raise SystemExit("正确性阈值必须满足 0 <= borderline-low <= threshold <= borderline-high <= 1。")

    if args.suite in {"qasper", "qasper-unanswerable"}:
        dataset = args.dataset or DEFAULT_QASPER
        cases, papers = load_qasper_cases(
            dataset,
            max_cases=max_cases,
            seed=args.seed,
            unanswerable_only=args.suite == "qasper-unanswerable",
        )
        chunks = build_qasper_chunks(papers)
    else:
        dataset = args.dataset or DEFAULT_SCHOLARQA_MULTI
        cases, chunks = load_scholarqa_multi_cases(dataset, max_cases=max_cases, seed=args.seed)
    if not cases:
        raise SystemExit(f"{args.suite} 没有筛选到可评测题目。")

    collection = args.collection or f"{args.suite.replace('-', '_')}_eval_v1"
    run_name = args.run_name or f"{args.suite.replace('-', '_')}_{len(cases)}"
    store = ChromaVectorStore(collection_name=collection)
    client = OpenAICompatibleClient()
    if args.rebuild:
        store.reset_collection()
    if store.count_chunks() == 0:
        print(f"正在为 {args.suite} 建立索引：{len(chunks)} chunks...", flush=True)

        def progress(indexed: int, total: int) -> None:
            print(f"  索引进度：{indexed}/{total}", flush=True)

        index_chunks_in_batches(store, chunks, client, progress_callback=progress)
    else:
        print(f"复用索引：collection={collection}, chunks={store.count_chunks()}", flush=True)

    print(f"开始 {args.suite} 测试：{len(cases)} 题，Top-{top_k}。", flush=True)
    results, summary = run_literature_evaluation(
        cases,
        store,
        client,
        top_k=top_k,
        retrieval_mode=args.retrieval_mode,
        correctness_threshold=threshold,
        correctness_borderline_low=borderline_low,
        correctness_borderline_high=borderline_high,
    )
    report = write_literature_report(results, summary, run_name=run_name)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"逐题结果：{report}", flush=True)


if __name__ == "__main__":
    main()

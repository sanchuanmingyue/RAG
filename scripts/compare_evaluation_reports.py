"""Print an A/B delta table for two retrieval evaluation JSON reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRICS = (
    ("Doc Recall@5", ("doc_recall_at_k",), "higher"),
    ("Section Recall@5", ("section_recall_at_k",), "higher"),
    ("Doc MRR", ("doc_mrr",), "higher"),
    ("Section MRR", ("section_mrr",), "higher"),
    ("P50 retrieval ms", ("retrieval_latency_ms", "p50"), "lower"),
    ("P95 retrieval ms", ("retrieval_latency_ms", "p95"), "lower"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two PaperReader-RAG evaluation reports.")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    return parser.parse_args()


def nested(data: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = data
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def main() -> None:
    args = parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    baseline_summary = baseline.get("summary") or {}
    candidate_summary = candidate.get("summary") or {}

    print(f"Baseline : {args.baseline}")
    print(f"Candidate: {args.candidate}")
    print(f"{'Metric':<24} {'Baseline':>12} {'Candidate':>12} {'Delta':>12}  Result")
    print("-" * 78)
    for label, path, direction in METRICS:
        old = nested(baseline_summary, path)
        new = nested(candidate_summary, path)
        if old is None or new is None:
            print(f"{label:<24} {'n/a':>12} {'n/a':>12} {'n/a':>12}")
            continue
        delta = new - old
        improved = delta > 0 if direction == "higher" else delta < 0
        unchanged = abs(delta) < 1e-12
        result = "same" if unchanged else ("better" if improved else "worse")
        print(f"{label:<24} {old:>12.4f} {new:>12.4f} {delta:>+12.4f}  {result}")

    keys = ("case_count", "top_k")
    mismatches = [
        key for key in keys if baseline_summary.get(key) != candidate_summary.get(key)
    ]
    baseline_config = baseline.get("config") or {}
    candidate_config = candidate.get("config") or {}
    for key in ("corpus_document_count", "chunk_size", "chunk_overlap"):
        if baseline_config.get(key) != candidate_config.get(key):
            mismatches.append(key)
    if mismatches:
        print("WARNING: comparison conditions differ: " + ", ".join(mismatches))


if __name__ == "__main__":
    main()

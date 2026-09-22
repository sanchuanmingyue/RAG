"""Evaluate the project-specific intent router on labeled JSONL cases."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.conversation import ConversationResolver
from backend.embeddings import OpenAICompatibleClient
from backend.memory import ConversationState
from backend.schemas import AgentIntent


INTENTS: tuple[AgentIntent, ...] = (
    "qa",
    "summary",
    "compare",
    "source_explain",
    "export",
    "library_status",
    "literature_search",
    "corpus_analysis",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PaperReader intent routing.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "intent_eval.jsonl",
        help="JSONL file containing query and expected_intent.",
    )
    parser.add_argument(
        "--subset",
        choices=("all", "single-turn", "contextual"),
        default="all",
        help="Evaluate all cases or only one routing subset.",
    )
    parser.add_argument(
        "--use-llm-fallback",
        action="store_true",
        help="Enable the production LLM fallback for ambiguous contextual turns (may consume quota).",
    )
    parser.add_argument("--output", type=Path, help="Optional result JSON path.")
    parser.add_argument("--min-accuracy", type=float, default=0.0)
    parser.add_argument("--min-macro-f1", type=float, default=0.0)
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} 不是合法 JSON。") from exc
        if item.get("expected_intent") not in INTENTS:
            raise ValueError(f"{path}:{line_number} 的 expected_intent 非法。")
        if not str(item.get("query") or "").strip():
            raise ValueError(f"{path}:{line_number} 缺少 query。")
        cases.append(item)
    if not cases:
        raise ValueError(f"评测集为空：{path}")
    return cases


def _state_from_case(case: dict[str, Any]) -> ConversationState:
    return ConversationState(
        last_intent=case.get("previous_intent"),
        last_user_query=str(case.get("previous_query") or ""),
        last_answer_summary=str(case.get("previous_answer_summary") or ""),
        active_artifact_type=case.get("active_artifact_type"),
    )


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(row["correct"] for row in rows)
    per_intent: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for intent in INTENTS:
        tp = sum(row["expected"] == intent and row["predicted"] == intent for row in rows)
        fp = sum(row["expected"] != intent and row["predicted"] == intent for row in rows)
        fn = sum(row["expected"] == intent and row["predicted"] != intent for row in rows)
        support = sum(row["expected"] == intent for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support:
            f1_values.append(f1)
        per_intent[intent] = {
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    confusion = Counter(
        (row["expected"], row["predicted"])
        for row in rows
        if not row["correct"]
    )
    contextual = [row for row in rows if row["is_contextual"]]
    followup_labeled = [row for row in rows if row["expected_is_followup"] is not None]
    return {
        "case_count": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "macro_f1": round(sum(f1_values) / len(f1_values), 4) if f1_values else 0.0,
        "contextual_accuracy": round(
            sum(row["correct"] for row in contextual) / len(contextual), 4
        ) if contextual else None,
        "followup_accuracy": round(
            sum(row["followup_correct"] for row in followup_labeled) / len(followup_labeled), 4
        ) if followup_labeled else None,
        "per_intent": per_intent,
        "confusions": [
            {"expected": expected, "predicted": predicted, "count": count}
            for (expected, predicted), count in confusion.most_common()
        ],
    }


def main() -> None:
    args = parse_args()
    cases = load_cases(args.dataset)
    if args.subset == "single-turn":
        cases = [case for case in cases if not case.get("previous_intent")]
    elif args.subset == "contextual":
        cases = [case for case in cases if case.get("previous_intent")]
    if not cases:
        raise SystemExit(f"subset={args.subset} 没有可评测样本。")

    llm_client = OpenAICompatibleClient() if args.use_llm_fallback else None
    resolver = ConversationResolver(llm_client)
    rows: list[dict[str, Any]] = []
    started = perf_counter()
    for case in cases:
        state = _state_from_case(case)
        resolution = resolver.resolve(str(case["query"]), state)
        expected = str(case["expected_intent"])
        expected_followup = case.get("expected_is_followup")
        rows.append(
            {
                "id": case.get("id"),
                "query": case["query"],
                "expected": expected,
                "predicted": resolution.intent,
                "correct": resolution.intent == expected,
                "is_contextual": bool(case.get("previous_intent")),
                "expected_is_followup": expected_followup,
                "predicted_is_followup": resolution.is_followup,
                "followup_correct": (
                    resolution.is_followup == expected_followup
                    if expected_followup is not None else None
                ),
                "followup_type": resolution.followup_type,
                "rewritten_query": resolution.rewritten_query,
            }
        )

    metrics = _metrics(rows)
    payload = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": str(args.dataset.resolve()),
        "subset": args.subset,
        "llm_fallback": args.use_llm_fallback,
        "latency_ms": round((perf_counter() - started) * 1000, 2),
        "metrics": metrics,
        "errors": [row for row in rows if not row["correct"] or row["followup_correct"] is False],
        "records": rows,
    }
    output = args.output or (
        PROJECT_ROOT
        / "storage"
        / "evals"
        / f"intent_router_{args.subset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if payload["errors"]:
        print("\n错误样本：")
        for row in payload["errors"]:
            print(f"- {row['id']}: {row['expected']} -> {row['predicted']} | {row['query']}")
    print(f"\n结果：{output}")

    if metrics["accuracy"] < args.min_accuracy or metrics["macro_f1"] < args.min_macro_f1:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

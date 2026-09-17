from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from backend.literature_evaluation import (
    evidence_recall,
    load_qasper_cases,
    load_scholarqa_multi_cases,
)


class LiteratureEvaluationTests(unittest.TestCase):
    def test_qasper_loader_separates_answerable_and_unanswerable_cases(self) -> None:
        fixture = {
            "paper-1": {
                "title": "Paper",
                "abstract": "Abstract text",
                "full_text": [{"section_name": "Methods", "paragraphs": ["Gold evidence paragraph."]}],
                "qas": [
                    {
                        "question": "What is used?",
                        "question_id": "answerable",
                        "answers": [{"answer": {
                            "unanswerable": False, "extractive_spans": ["A method"],
                            "yes_no": None, "free_form_answer": "", "evidence": ["Gold evidence paragraph."],
                        }}],
                    },
                    {
                        "question": "What is missing?",
                        "question_id": "unanswerable",
                        "answers": [{"answer": {
                            "unanswerable": True, "extractive_spans": [],
                            "yes_no": None, "free_form_answer": "", "evidence": [],
                        }}],
                    },
                ],
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qasper-dev-v0.3.json"
            path.write_text(json.dumps(fixture), encoding="utf-8")
            answerable, _ = load_qasper_cases(path, max_cases=30)
            unanswerable, _ = load_qasper_cases(path, max_cases=15, unanswerable_only=True)

        self.assertEqual([case.case_id for case in answerable], ["answerable"])
        self.assertEqual(answerable[0].gold_evidence, ("Gold evidence paragraph.",))
        self.assertEqual([case.case_id for case in unanswerable], ["unanswerable"])
        self.assertTrue(unanswerable[0].unanswerable)

    def test_evidence_recall_uses_final_generation_context(self) -> None:
        hits = [{
            "text": "representative text",
            "generation_text": "The model is trained with gold evidence paragraph and regularization.",
            "metadata": {},
        }]
        self.assertEqual(evidence_recall(["gold evidence paragraph"], hits), 1.0)
        self.assertEqual(evidence_recall(["completely absent facts"], hits), 0.0)

    def test_scholarqa_multi_loader_scopes_each_question_to_its_sources(self) -> None:
        fixture = [{
            "id": "multi-1",
            "input": "Compare the approaches.",
            "output": "They differ in training [0] [1].",
            "ctxs": [
                {"title": "Paper A", "text": "Approach A."},
                {"title": "Paper B", "text": "Approach B."},
            ],
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "human_answers.json"
            path.write_text(json.dumps(fixture), encoding="utf-8")
            cases, chunks = load_scholarqa_multi_cases(path, max_cases=10)

        self.assertEqual(len(cases), 1)
        self.assertEqual(len(cases[0].paper_ids), 2)
        self.assertEqual({chunk.paper_id for chunk in chunks}, set(cases[0].paper_ids))


if __name__ == "__main__":
    unittest.main()

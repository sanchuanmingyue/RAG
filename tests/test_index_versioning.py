"""Regression checks for persisted chunking-version migration."""

from __future__ import annotations

import unittest

from backend.text_splitter import SPLITTER_VERSION
from backend.vector_store import ChromaVectorStore


class _FakeCollection:
    def get(self, *, include):
        return {
            "metadatas": [
                {"paper_id": "legacy", "file_name": "legacy.pdf"},
                {
                    "paper_id": "current",
                    "file_name": "current.pdf",
                    "splitter_version": SPLITTER_VERSION,
                },
            ]
        }


class IndexVersioningTests(unittest.TestCase):
    def test_list_papers_marks_only_legacy_chunks_outdated(self) -> None:
        store = ChromaVectorStore.__new__(ChromaVectorStore)
        store.collection = _FakeCollection()
        store._paper_list_cache = None

        papers = {paper["paper_id"]: paper for paper in store.list_papers()}

        self.assertTrue(papers["legacy"]["index_outdated"])
        self.assertEqual(papers["legacy"]["splitter_version"], "legacy")
        self.assertFalse(papers["current"]["index_outdated"])
        self.assertEqual(papers["current"]["splitter_version"], SPLITTER_VERSION)


if __name__ == "__main__":
    unittest.main()

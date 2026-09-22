"""Checks for cross-page, section-aware chunking."""

from __future__ import annotations

import unittest

from backend.text_splitter import SPLITTER_VERSION, split_pages, split_text


class TextSplitterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pages = [
            {
                "paper_id": "paper",
                "file_name": "paper.pdf",
                "page": 1,
                "text": (
                    "1 Introduction\n"
                    "The first page explains the research problem in detail. "
                    "Its final sentence leads into the next page."
                ),
            },
            {
                "paper_id": "paper",
                "file_name": "paper.pdf",
                "page": 2,
                "text": (
                    "The introduction continues with motivation and prior work.\n\n"
                    "2 Method\nThe proposed method begins with retrieval."
                ),
            },
            {
                "paper_id": "paper",
                "file_name": "paper.pdf",
                "page": 3,
                "text": "The proposed method then reranks the retrieved evidence.",
            },
        ]

    def test_cross_page_sections_share_parent_and_page_range(self) -> None:
        chunks = split_pages(
            self.pages,
            chunk_size=120,
            chunk_overlap=20,
        )
        introduction = [chunk for chunk in chunks if chunk.section_type == "introduction"]
        method = [chunk for chunk in chunks if chunk.section_type == "method"]

        self.assertTrue(introduction)
        self.assertTrue(method)
        self.assertEqual(len({chunk.parent_id for chunk in introduction}), 1)
        self.assertEqual(len({chunk.parent_id for chunk in method}), 1)
        self.assertTrue(any(chunk.page_start == 1 and chunk.page_end == 2 for chunk in introduction))
        self.assertTrue(any(chunk.page_start == 2 and chunk.page_end == 3 for chunk in method))
        self.assertTrue(any(chunk.page_numbers == "1,2" for chunk in introduction))

    def test_paragraph_strategy_prefers_sentence_boundaries(self) -> None:
        text = (
            "This is the first complete sentence. "
            "This is the second complete sentence. "
            "This is the final complete sentence."
        )
        semantic = split_text(text, chunk_size=72, chunk_overlap=12)

        self.assertTrue(semantic[0].endswith("sentence."))

    def test_composite_experiment_heading_is_recognized(self) -> None:
        pages = [
            {
                "paper_id": "paper",
                "file_name": "paper.pdf",
                "page": 7,
                "text": (
                    "5. Experimental simulation and evaluation\n"
                    "5.1. Environment settings\n"
                    "We consider a cloud-edge network topology.\n"
                    "5.2. Performance evaluation\n"
                    "The proposed method outperforms the baselines."
                ),
            }
        ]

        chunks = split_pages(pages)

        self.assertTrue(chunks)
        self.assertTrue(all(chunk.section_type == "experiment" for chunk in chunks))
        self.assertTrue(any("Experimental simulation and evaluation" in chunk.section_path for chunk in chunks))

    def test_chunk_metadata_contains_splitter_version(self) -> None:
        chunk = split_pages(self.pages)[0]

        self.assertEqual(chunk.to_metadata()["splitter_version"], SPLITTER_VERSION)


if __name__ == "__main__":
    unittest.main()

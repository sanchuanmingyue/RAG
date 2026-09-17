"""Chroma vector store wrapper with hybrid retrieval support."""

from __future__ import annotations

import math
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass
from hashlib import sha1
from time import perf_counter
from typing import Any, Callable

import chromadb

from backend.config import VECTOR_DB_DIR, settings
from backend.embeddings import OpenAICompatibleClient
from backend.reranker import SectionReranker
from backend.text_splitter import TextChunk


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Tokenize English words and Chinese character/bigram terms for BM25."""

    tokens: list[str] = []
    for matched in TOKEN_PATTERN.findall(text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", matched):
            tokens.extend(matched)
            tokens.extend(matched[index : index + 2] for index in range(len(matched) - 1))
        else:
            tokens.append(matched)
    return tokens


@dataclass
class _KeywordRow:
    row_id: str
    text: str
    metadata: dict[str, Any]
    token_counts: Counter[str]


class ChromaVectorStore:
    """Small Chroma facade used by the Streamlit app and Agent tools."""

    def __init__(self, collection_name: str | None = None) -> None:
        VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        self.collection_name = collection_name or settings.chroma_collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "PaperReader-RAG chunks"},
        )
        self.section_collection_name = f"{self.collection_name}_sections"
        self.section_collection = self.client.get_or_create_collection(
            name=self.section_collection_name,
            metadata={"description": "PaperReader-RAG section centroids"},
        )
        self._keyword_rows: list[_KeywordRow] | None = None
        self._keyword_postings: dict[str, list[tuple[int, int]]] = {}
        self._keyword_doc_frequency: Counter[str] = Counter()
        self._keyword_query_cache: OrderedDict[str, dict[int, float]] = OrderedDict()
        self._paper_list_cache: list[dict[str, Any]] | None = None
        self.section_reranker = SectionReranker()
        self.last_query_timings_ms: dict[str, float] = {}
        self._last_vector_timings_ms: dict[str, float] = {}
        self._last_keyword_timings_ms: dict[str, float] = {}

    def index_chunks(
        self,
        chunks: list[TextChunk],
        embedding_client: OpenAICompatibleClient,
        *,
        replace_existing: bool = False,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> int:
        if not chunks:
            return 0

        documents = [chunk.text for chunk in chunks]
        embedding_inputs = [chunk.to_embedding_text(settings.embedding_section_context) for chunk in chunks]
        if progress_callback is None:
            embeddings = embedding_client.embed_texts(embedding_inputs)
        else:
            embeddings = embedding_client.embed_texts(
                embedding_inputs,
                progress_callback=lambda current, total: progress_callback("embedding", current, total),
            )

        if replace_existing:
            paper_ids = {chunk.paper_id for chunk in chunks if chunk.paper_id}
            if len(paper_ids) != 1:
                raise ValueError("replace_existing requires chunks from exactly one paper")
            # Embed first so a temporary model failure leaves the existing index
            # intact. Deleting just before upsert also removes stale tail chunks
            # when a revised PDF produces fewer chunks than its previous version.
            replaced_paper_id = next(iter(paper_ids))
            self.collection.delete(where={"paper_id": replaced_paper_id})
            if hasattr(self, "section_collection"):
                self.section_collection.delete(where={"paper_id": replaced_paper_id})

        if progress_callback is not None:
            progress_callback("writing", 0, len(chunks))
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=documents,
            embeddings=embeddings,
            metadatas=[chunk.to_metadata() for chunk in chunks],
        )
        if settings.enable_section_index and hasattr(self, "section_collection"):
            self._upsert_section_records(chunks, embeddings)
        self._invalidate_keyword_index()
        if progress_callback is not None:
            progress_callback("writing", len(chunks), len(chunks))
        return len(chunks)

    def _upsert_section_records(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> None:
        """Maintain one centroid vector and one representative document per section."""

        grouped: dict[str, list[tuple[TextChunk, list[float]]]] = {}
        for chunk, embedding in zip(chunks, embeddings):
            key = self._chunk_section_key(chunk)
            grouped.setdefault(key, []).append((chunk, embedding))
        if not grouped:
            return

        ids = [self._section_record_id(key) for key in grouped]
        existing = self.section_collection.get(ids=ids, include=["documents", "metadatas", "embeddings"])
        existing_embeddings = existing.get("embeddings")
        if existing_embeddings is None:
            existing_embeddings = []
        existing_by_id: dict[str, tuple[str, dict[str, Any], list[float]]] = {}
        for row_id, document, metadata, embedding in zip(
            existing.get("ids", []),
            existing.get("documents") or [],
            existing.get("metadatas") or [],
            existing_embeddings,
        ):
            existing_by_id[str(row_id)] = (str(document or ""), dict(metadata or {}), list(embedding))

        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        centroids: list[list[float]] = []
        for key, values in grouped.items():
            row_id = self._section_record_id(key)
            chunk_count = len(values)
            dimension = len(values[0][1])
            centroid = [sum(value[1][index] for value in values) / chunk_count for index in range(dimension)]
            chunk = values[0][0]
            metadata = chunk.to_metadata()
            metadata["section_index_id"] = row_id
            metadata["section_chunk_count"] = chunk_count
            heading = chunk.section_path or chunk.section_title or f"Section {chunk.parent_index}"
            document = f"Section: {heading}\n\n" + "\n\n".join(value[0].text for value in values)

            previous = existing_by_id.get(row_id)
            if previous is not None:
                old_document, old_metadata, old_centroid = previous
                old_count = max(int(old_metadata.get("section_chunk_count") or 0), 0)
                total_count = old_count + chunk_count
                if old_count and len(old_centroid) == dimension:
                    centroid = [
                        (old_centroid[index] * old_count + centroid[index] * chunk_count) / total_count
                        for index in range(dimension)
                    ]
                metadata = {**old_metadata, **metadata, "section_chunk_count": total_count}
                document = f"{old_document}\n\n{document}" if old_document else document

            documents.append(document[: max(settings.section_document_max_chars, 256)])
            metadatas.append(metadata)
            centroids.append(centroid)

        self.section_collection.upsert(ids=ids, documents=documents, embeddings=centroids, metadatas=metadatas)

    @staticmethod
    def _chunk_section_key(chunk: TextChunk) -> str:
        section_id = chunk.benchmark_section_id or chunk.parent_id or chunk.section_path or chunk.chunk_id
        return f"{chunk.paper_id}::{section_id}"

    @staticmethod
    def _section_record_id(section_key: str) -> str:
        return f"section_{sha1(section_key.encode('utf-8')).hexdigest()}"

    def query(
        self,
        query_text: str,
        embedding_client: OpenAICompatibleClient,
        top_k: int = 5,
        paper_id: str | None = None,
        paper_ids: list[str] | None = None,
        section_types: list[str] | None = None,
        *,
        retrieval_mode: str | None = None,
        expand_parent: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve documents first, then rank aggregated sections within them."""

        started = perf_counter()
        # Timings are per request.  Clear the unused modality so a keyword-only
        # request cannot accidentally report timings from a preceding hybrid one.
        self._last_vector_timings_ms = {}
        self._last_keyword_timings_ms = {}
        mode = (retrieval_mode or settings.retrieval_mode or "hybrid").lower()
        rerank_ms = 0.0
        document_stage_ms = 0.0
        selected_paper_ids: list[str] | None = (
            [paper_id] if paper_id else list(dict.fromkeys(paper_ids or [])) or None
        )
        if settings.enable_two_stage_retrieval and selected_paper_ids is None:
            document_stage_started = perf_counter()
            document_hits = self._retrieve_candidates(
                query_text,
                embedding_client,
                mode=mode,
                top_k=max(top_k, settings.document_candidate_k),
            )
            if settings.enable_reranker:
                rerank_started = perf_counter()
                document_hits = self._rerank(query_text, document_hits)
                rerank_ms += (perf_counter() - rerank_started) * 1000
            selected_paper_ids = self._rank_documents(document_hits, settings.document_top_k)
            document_stage_ms = (perf_counter() - document_stage_started) * 1000

        section_started = perf_counter()
        candidate_k = max(top_k, settings.section_candidate_k if settings.enable_two_stage_retrieval else settings.hybrid_candidate_k)
        hits = self._retrieve_candidates(
            query_text,
            embedding_client,
            mode=mode,
            top_k=candidate_k,
            paper_ids=selected_paper_ids,
            section_types=section_types,
        )
        if settings.enable_reranker:
            rerank_started = perf_counter()
            hits = self._rerank(query_text, hits)
            rerank_ms += (perf_counter() - rerank_started) * 1000
        section_index_hits: list[dict[str, Any]] = []
        if (
            settings.enable_section_index
            and mode != "keyword"
            and hasattr(self, "section_collection")
            and self.section_collection.count() > 0
        ):
            section_index_hits = self._section_vector_query(
                query_text,
                embedding_client,
                top_k=candidate_k,
                paper_ids=selected_paper_ids,
                section_types=section_types,
            )
            if settings.enable_reranker:
                rerank_started = perf_counter()
                section_index_hits = self._rerank(query_text, section_index_hits)
                rerank_ms += (perf_counter() - rerank_started) * 1000
        section_stage_ms = (perf_counter() - section_started) * 1000

        diversity_started = perf_counter()
        if settings.enable_two_stage_retrieval:
            hits = self._aggregate_sections(query_text, hits, settings.section_support_chunks)
            hits = self._merge_section_candidates(query_text, hits, section_index_hits)
        elif settings.enable_section_diversity:
            hits = self._diversify_sections(hits, max_per_section=settings.section_max_chunks)
        diversity_ms = (perf_counter() - diversity_started) * 1000
        section_reranker_started = perf_counter()
        if settings.enable_section_reranker and hasattr(self, "section_reranker"):
            hits = self.section_reranker.rerank(query_text, hits)
        section_reranker_ms = (perf_counter() - section_reranker_started) * 1000
        hits = hits[:top_k]
        parent_started = perf_counter()
        if settings.enable_parent_context if expand_parent is None else expand_parent:
            hits = self.expand_parent_context(hits)
        parent_context_ms = (perf_counter() - parent_started) * 1000
        self.last_query_timings_ms = {
            "embedding_ms": round(self._last_vector_timings_ms.get("embedding_ms", 0.0), 2),
            "vector_db_ms": round(self._last_vector_timings_ms.get("vector_db_ms", 0.0), 2),
            "section_vector_db_ms": round(self._last_vector_timings_ms.get("section_vector_db_ms", 0.0), 2),
            "keyword_index_build_ms": round(self._last_keyword_timings_ms.get("index_build_ms", 0.0), 2),
            "keyword_scoring_ms": round(self._last_keyword_timings_ms.get("scoring_ms", 0.0), 2),
            "keyword_cache_hits": int(self._last_keyword_timings_ms.get("cache_hits", 0.0)),
            "document_stage_ms": round(document_stage_ms, 2),
            "section_stage_ms": round(section_stage_ms, 2),
            "rerank_ms": round(rerank_ms, 2),
            "section_diversity_ms": round(diversity_ms, 2),
            "section_reranker_ms": round(section_reranker_ms, 2),
            "cross_encoder_ms": round(section_reranker_ms, 2),
            "parent_context_ms": round(parent_context_ms, 2),
            "total_ms": round((perf_counter() - started) * 1000, 2),
        }
        return hits

    def _section_vector_query(
        self,
        query_text: str,
        embedding_client: OpenAICompatibleClient,
        *,
        top_k: int,
        paper_ids: list[str] | None,
        section_types: list[str] | None,
    ) -> list[dict[str, Any]]:
        embedding_started = perf_counter()
        query_embedding = embedding_client.embed_query(query_text)
        self._last_vector_timings_ms["embedding_ms"] = self._last_vector_timings_ms.get("embedding_ms", 0.0) + (
            perf_counter() - embedding_started
        ) * 1000
        query_started = perf_counter()
        result = self.section_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=self._build_where(paper_ids=paper_ids, section_types=section_types),
            include=["documents", "metadatas", "distances"],
        )
        self._last_vector_timings_ms["section_vector_db_ms"] = self._last_vector_timings_ms.get(
            "section_vector_db_ms", 0.0
        ) + (perf_counter() - query_started) * 1000
        hits = self._normalize_query_result(result)
        for rank, hit in enumerate(hits, start=1):
            hit["section_vector_rank"] = rank
            hit["retrieval_source"] = "section_vector"
            hit["section_document"] = hit.get("text", "")
        return hits

    def _retrieve_candidates(
        self,
        query_text: str,
        embedding_client: OpenAICompatibleClient,
        *,
        mode: str,
        top_k: int,
        paper_ids: list[str] | None = None,
        section_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not paper_ids and paper_ids is not None:
            return []
        if mode == "keyword":
            return self.keyword_query(query_text, top_k=top_k, paper_ids=paper_ids, section_types=section_types)
        if mode == "vector":
            return self.vector_query(
                query_text, embedding_client, top_k=top_k, paper_ids=paper_ids, section_types=section_types
            )
        vector_hits = self.vector_query(
            query_text, embedding_client, top_k=top_k, paper_ids=paper_ids, section_types=section_types
        )
        keyword_hits = self.keyword_query(
            query_text, top_k=top_k, paper_ids=paper_ids, section_types=section_types
        )
        return self._rrf_fuse(vector_hits, keyword_hits)

    def vector_query(
        self,
        query_text: str,
        embedding_client: OpenAICompatibleClient,
        top_k: int = 5,
        paper_id: str | None = None,
        paper_ids: list[str] | None = None,
        section_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        embedding_started = perf_counter()
        query_embedding = embedding_client.embed_query(query_text)
        embedding_ms = (perf_counter() - embedding_started) * 1000
        vector_db_started = perf_counter()
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=self._build_where(paper_id=paper_id, paper_ids=paper_ids, section_types=section_types),
            include=["documents", "metadatas", "distances"],
        )
        self._last_vector_timings_ms["embedding_ms"] = self._last_vector_timings_ms.get("embedding_ms", 0.0) + embedding_ms
        self._last_vector_timings_ms["vector_db_ms"] = self._last_vector_timings_ms.get("vector_db_ms", 0.0) + (
            perf_counter() - vector_db_started
        ) * 1000
        hits = self._normalize_query_result(result)
        for rank, hit in enumerate(hits, start=1):
            hit["vector_rank"] = rank
            hit["retrieval_source"] = "vector"
        return hits

    def keyword_query(
        self,
        query_text: str,
        top_k: int = 20,
        paper_id: str | None = None,
        paper_ids: list[str] | None = None,
        section_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        query_tokens = _tokenize(query_text)
        if not query_tokens:
            return []

        index_build_started = perf_counter()
        self._ensure_keyword_index()
        index_build_ms = (perf_counter() - index_build_started) * 1000
        scoring_started = perf_counter()
        cache_key = " ".join(query_text.lower().split())
        cache = getattr(self, "_keyword_query_cache", None)
        scored_by_row = cache.pop(cache_key, None) if cache is not None else None
        cache_hit = scored_by_row is not None
        if scored_by_row is None:
            scored_by_row = self._score_keyword_rows(query_text, query_tokens)
        if cache is not None and settings.keyword_query_cache_size > 0:
            cache[cache_key] = scored_by_row
            while len(cache) > settings.keyword_query_cache_size:
                cache.popitem(last=False)

        scored: list[dict[str, Any]] = []
        for row_index, score in scored_by_row.items():
            row = (self._keyword_rows or [])[row_index]
            if not self._matches_filter(row.metadata, paper_id, paper_ids, section_types):
                continue
            scored.append(
                {
                    "text": row.text,
                    "metadata": row.metadata,
                    "distance": None,
                    "keyword_score": score,
                    "retrieval_source": "keyword",
                }
            )

        self._last_keyword_timings_ms["index_build_ms"] = self._last_keyword_timings_ms.get("index_build_ms", 0.0) + index_build_ms
        self._last_keyword_timings_ms["scoring_ms"] = self._last_keyword_timings_ms.get("scoring_ms", 0.0) + (
            perf_counter() - scoring_started
        ) * 1000
        self._last_keyword_timings_ms["cache_hits"] = self._last_keyword_timings_ms.get("cache_hits", 0.0) + float(cache_hit)

        scored.sort(key=lambda hit: hit.get("keyword_score", 0.0), reverse=True)
        for rank, hit in enumerate(scored[:top_k], start=1):
            hit["keyword_rank"] = rank
        return scored[:top_k]

    def _score_keyword_rows(self, query_text: str, query_tokens: list[str]) -> dict[int, float]:
        query_counter = Counter(query_tokens)
        scored_by_row: dict[int, float] = {}
        total_docs = max(len(self._keyword_rows or []), 1)
        for token, query_frequency in query_counter.items():
            postings = self._keyword_postings.get(token, [])
            idf = math.log((total_docs + 1) / (self._keyword_doc_frequency[token] + 1)) + 1.0
            for row_index, term_frequency in postings:
                scored_by_row[row_index] = scored_by_row.get(row_index, 0.0) + (
                    (1.0 + math.log(term_frequency)) * idf * query_frequency
                )
        query_phrase = query_text.strip().lower()
        if query_phrase:
            for row_index in list(scored_by_row):
                row = (self._keyword_rows or [])[row_index]
                if query_phrase in row.text.lower():
                    scored_by_row[row_index] += 2.0
        return scored_by_row

    def expand_parent_context(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        expanded: list[dict[str, Any]] = []
        seen_parent_ids: set[str] = set()
        for hit in hits:
            metadata = hit.get("metadata", {})
            parent_id = str(metadata.get("parent_id") or "")
            if not parent_id or parent_id in seen_parent_ids:
                expanded.append(hit)
                continue
            seen_parent_ids.add(parent_id)
            result = self.collection.get(
                where={"parent_id": parent_id},
                include=["documents", "metadatas"],
            )
            rows = self._normalize_get_result(result)
            if not rows:
                expanded.append(hit)
                continue
            rows.sort(key=lambda row: int(row.get("metadata", {}).get("child_index") or 0))
            parent_text = "\n".join(row.get("text", "") for row in rows if row.get("text"))
            parent_pages = sorted(
                {
                    str(row.get("metadata", {}).get("page", ""))
                    for row in rows
                    if row.get("metadata", {}).get("page", "") != ""
                }
            )
            parent_hit = dict(hit)
            parent_hit["text"] = parent_text or hit.get("text", "")
            parent_metadata = dict(metadata)
            parent_metadata["expanded_from_child_id"] = metadata.get("chunk_id", "")
            parent_metadata["parent_pages"] = ", ".join(parent_pages)
            parent_hit["metadata"] = parent_metadata
            parent_hit["is_parent_context"] = True
            expanded.append(parent_hit)
        return expanded

    def list_papers(self) -> list[dict[str, Any]]:
        if self._paper_list_cache is not None:
            return [dict(item) for item in self._paper_list_cache]
        result = self.collection.get(include=["metadatas"])
        papers: dict[str, dict[str, Any]] = {}

        for metadata in result.get("metadatas", []):
            if not metadata:
                continue

            paper_id = str(metadata.get("paper_id", ""))
            if not paper_id:
                continue

            papers.setdefault(
                paper_id,
                {
                    "paper_id": paper_id,
                    "file_name": metadata.get("file_name", ""),
                    "chunk_count": 0,
                },
            )
            papers[paper_id]["chunk_count"] += 1

        self._paper_list_cache = sorted(papers.values(), key=lambda item: item["file_name"])
        return [dict(item) for item in self._paper_list_cache]

    def count_chunks(self, paper_id: str | None = None) -> int:
        if paper_id is None:
            return self.collection.count()
        result = self.collection.get(where={"paper_id": paper_id}, include=[])
        return len(result.get("ids", []))

    def peek_chunks(self, limit: int = 5, paper_id: str | None = None) -> list[dict[str, Any]]:
        result = self.collection.get(
            limit=limit,
            where={"paper_id": paper_id} if paper_id else None,
            include=["documents", "metadatas"],
        )
        return self._normalize_get_result(result)

    def get_paper_chunks(self, paper_id: str, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        result = self.collection.get(
            where={"paper_id": paper_id},
            limit=limit,
            offset=offset,
            include=["documents", "metadatas"],
        )
        return self._normalize_get_result(result)

    def delete_paper(self, paper_id: str) -> None:
        self.collection.delete(where={"paper_id": paper_id})
        self.section_collection.delete(where={"paper_id": paper_id})
        self._invalidate_keyword_index()

    def reset_collection(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.client.delete_collection(self.section_collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "PaperReader-RAG chunks"},
        )
        self.section_collection = self.client.get_or_create_collection(
            name=self.section_collection_name,
            metadata={"description": "PaperReader-RAG section centroids"},
        )
        self._invalidate_keyword_index()

    def warmup(self) -> dict[str, Any]:
        """Build CPU-side indexes and load the optional reranker before traffic."""

        started = perf_counter()
        self._ensure_keyword_index()
        reranker_ready = self.section_reranker.warmup()
        return {
            "keyword_rows": len(self._keyword_rows or []),
            "section_count": self.section_collection.count(),
            "section_reranker_ready": reranker_ready,
            "section_reranker": self.section_reranker.status,
            "warmup_ms": round((perf_counter() - started) * 1000, 2),
        }

    def explain_hit(self, hit: dict[str, Any]) -> dict[str, Any]:
        metadata = hit.get("metadata", {})
        return {
            "paper_id": metadata.get("paper_id", ""),
            "file_name": metadata.get("file_name", ""),
            "page": metadata.get("page", ""),
            "chunk_id": metadata.get("chunk_id", ""),
            "parent_id": metadata.get("parent_id", ""),
            "section_title": metadata.get("section_title", ""),
            "subsection_title": metadata.get("subsection_title", ""),
            "section_type": metadata.get("section_type", ""),
            "section_path": metadata.get("section_path", ""),
            "retrieval_source": hit.get("retrieval_source", ""),
            "rrf_score": hit.get("rrf_score"),
            "rerank_score": hit.get("rerank_score"),
            "keyword_score_normalized": hit.get("keyword_score_normalized"),
            "vector_score_normalized": hit.get("vector_score_normalized"),
            "reference_penalty_applied": hit.get("reference_penalty_applied"),
            "section_score": hit.get("section_score"),
            "section_chunk_count": hit.get("section_chunk_count"),
            "section_query_coverage": hit.get("section_query_coverage"),
            "distance": hit.get("distance"),
            "text": hit.get("text", ""),
        }

    def _build_where(
        self,
        paper_id: str | None = None,
        paper_ids: list[str] | None = None,
        section_types: list[str] | None = None,
    ) -> dict[str, Any] | None:
        where_clauses: list[dict[str, Any]] = []
        if paper_id:
            where_clauses.append({"paper_id": paper_id})
        elif paper_ids:
            cleaned_paper_ids = list(dict.fromkeys(value for value in paper_ids if value))
            if cleaned_paper_ids:
                where_clauses.append({"paper_id": {"$in": cleaned_paper_ids}})
        if section_types:
            cleaned_section_types = [section_type for section_type in section_types if section_type]
            if cleaned_section_types:
                where_clauses.append({"section_type": {"$in": cleaned_section_types}})

        if len(where_clauses) == 1:
            return where_clauses[0]
        if len(where_clauses) > 1:
            return {"$and": where_clauses}
        return None

    def _rrf_fuse(self, vector_hits: list[dict[str, Any]], keyword_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}

        def add_hits(hits: list[dict[str, Any]], rank_key: str, source_name: str) -> None:
            for rank, hit in enumerate(hits, start=1):
                metadata = hit.get("metadata", {})
                chunk_id = str(metadata.get("chunk_id") or metadata.get("id") or hash(hit.get("text", "")))
                current = fused.setdefault(chunk_id, dict(hit))
                current["metadata"] = metadata
                current["text"] = hit.get("text", current.get("text", ""))
                current[rank_key] = rank
                current["rrf_score"] = current.get("rrf_score", 0.0) + 1.0 / (settings.rrf_k + rank)
                sources = set(str(current.get("retrieval_source", "")).split("+")) if current.get("retrieval_source") else set()
                sources.add(source_name)
                current["retrieval_source"] = "+".join(sorted(sources))
                if hit.get("distance") is not None:
                    current["distance"] = hit.get("distance")
                if hit.get("keyword_score") is not None:
                    current["keyword_score"] = hit.get("keyword_score")

        add_hits(vector_hits, "vector_rank", "vector")
        add_hits(keyword_hits, "keyword_rank", "keyword")

        merged = list(fused.values())
        merged.sort(key=lambda hit: hit.get("rrf_score", 0.0), reverse=True)
        return merged

    def _ensure_keyword_index(self) -> None:
        if self._keyword_rows is not None:
            return

        result = self.collection.get(include=["documents", "metadatas"])
        rows = self._normalize_get_result(result)
        keyword_rows: list[_KeywordRow] = []
        postings: dict[str, list[tuple[int, int]]] = {}
        document_frequency: Counter[str] = Counter()
        for row in rows:
            token_counts = Counter(_tokenize(row.get("text", "")))
            row_index = len(keyword_rows)
            keyword_rows.append(
                _KeywordRow(
                    row_id=str(row.get("id") or ""),
                    text=str(row.get("text") or ""),
                    metadata=dict(row.get("metadata") or {}),
                    token_counts=token_counts,
                )
            )
            for token, count in token_counts.items():
                postings.setdefault(token, []).append((row_index, count))
            document_frequency.update(token_counts.keys())

        self._keyword_rows = keyword_rows
        self._keyword_postings = postings
        self._keyword_doc_frequency = document_frequency

    def _invalidate_keyword_index(self) -> None:
        self._keyword_rows = None
        self._keyword_postings = {}
        self._keyword_doc_frequency = Counter()
        self._keyword_query_cache = OrderedDict()
        self._paper_list_cache = None

    @staticmethod
    def _matches_filter(
        metadata: dict[str, Any],
        paper_id: str | None,
        paper_ids: list[str] | None,
        section_types: list[str] | None,
    ) -> bool:
        if paper_id and metadata.get("paper_id") != paper_id:
            return False
        if not paper_id and paper_ids and metadata.get("paper_id") not in set(paper_ids):
            return False
        if section_types and metadata.get("section_type") not in set(section_types):
            return False
        return True

    def _rerank(self, query_text: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        query_tokens = set(_tokenize(query_text))
        if not query_tokens:
            return hits

        keyword_scores = [max(float(hit.get("keyword_score") or 0.0), 0.0) for hit in hits]
        vector_scores = [
            1.0 / (1.0 + max(float(hit["distance"]), 0.0))
            if isinstance(hit.get("distance"), (int, float))
            else 0.0
            for hit in hits
        ]
        rrf_scores = [max(float(hit.get("rrf_score") or 0.0), 0.0) for hit in hits]
        keyword_max = max(keyword_scores, default=0.0)
        vector_min, vector_max = min(vector_scores, default=0.0), max(vector_scores, default=0.0)
        rrf_max = max(rrf_scores, default=0.0)
        asks_for_references = self._asks_for_references(query_text)

        for index, hit in enumerate(hits):
            text_tokens = _tokenize(hit.get("text", ""))
            text_set = set(text_tokens)
            overlap = len(query_tokens & text_set) / max(len(query_tokens), 1)
            metadata = hit.get("metadata", {})
            heading_tokens = set(
                _tokenize(
                    " ".join(
                        str(metadata.get(key) or "")
                        for key in ("section_path", "section_title", "subsection_title", "section_type")
                    )
                )
            )
            heading_overlap = len(query_tokens & heading_tokens) / max(len(query_tokens), 1)
            phrase_bonus = 0.1 if query_text.lower().strip() in hit.get("text", "").lower() else 0.0
            source_bonus = 0.03 if hit.get("retrieval_source") == "keyword+vector" else 0.0
            keyword_normalized = keyword_scores[index] / keyword_max if keyword_max else 0.0
            if vector_max > vector_min:
                vector_normalized = (vector_scores[index] - vector_min) / (vector_max - vector_min)
            else:
                vector_normalized = 1.0 if vector_scores[index] else 0.0
            rrf_normalized = rrf_scores[index] / rrf_max if rrf_max else 0.0
            reference_penalty = (
                settings.reference_section_penalty
                if not asks_for_references and self._is_reference_hit(hit)
                else 0.0
            )
            hit["section_heading_score"] = heading_overlap
            hit["keyword_score_normalized"] = keyword_normalized
            hit["vector_score_normalized"] = vector_normalized
            hit["rrf_score_normalized"] = rrf_normalized
            hit["reference_penalty_applied"] = reference_penalty
            hit["rerank_score"] = (
                0.3 * overlap
                + settings.section_heading_weight * heading_overlap
                + phrase_bonus
                + source_bonus
                + 0.05 * rrf_normalized
                + 0.2 * keyword_normalized
                + 0.2 * vector_normalized
                - reference_penalty
            )

        hits.sort(key=lambda hit: hit.get("rerank_score", 0.0), reverse=True)
        return hits

    @staticmethod
    def _asks_for_references(query_text: str) -> bool:
        lowered = query_text.lower()
        return bool(
            re.search(r"\b(references?|bibliograph(?:y|ies)|citations?|cited)\b", lowered)
            or any(term in query_text for term in ("参考文献", "文献列表", "引用了", "引用哪些", "出处"))
        )

    @staticmethod
    def _is_reference_hit(hit: dict[str, Any]) -> bool:
        metadata = hit.get("metadata", {})
        if str(metadata.get("section_type") or "").lower() == "references":
            return True
        heading = " ".join(
            str(metadata.get(key) or "") for key in ("section_title", "subsection_title", "section_path")
        )
        if re.search(r"\b(references?|bibliography)\b", heading, flags=re.I):
            return True
        text_prefix = str(hit.get("text") or "").lstrip()[:160]
        return bool(
            re.match(
                r"^(?:#{1,6}\s*)?(?:(?:[ivxlcdm]+|\d+(?:\.\d+)*)[.)\s:-]+)?(?:references?|bibliography)\b",
                text_prefix,
                flags=re.I,
            )
        )

    @staticmethod
    def _rank_documents(hits: list[dict[str, Any]], top_k: int) -> list[str]:
        grouped: dict[str, list[tuple[int, float]]] = {}
        for rank, hit in enumerate(hits, start=1):
            metadata = hit.get("metadata", {})
            paper_id = str(metadata.get("paper_id") or metadata.get("benchmark_doc_id") or "")
            if paper_id:
                grouped.setdefault(paper_id, []).append((rank, float(hit.get("rerank_score") or 0.0)))

        scores: list[tuple[str, float]] = []
        for paper_id, values in grouped.items():
            ordered = sorted(values, key=lambda value: value[1], reverse=True)[:3]
            best = ordered[0][1]
            mean_score = sum(value[1] for value in ordered) / len(ordered)
            first_rank = min(value[0] for value in values)
            scores.append((paper_id, 0.65 * best + 0.25 * mean_score + 0.1 / first_rank))
        scores.sort(key=lambda value: value[1], reverse=True)
        return [paper_id for paper_id, _score in scores[: max(top_k, 1)]]

    @staticmethod
    def _section_key(hit: dict[str, Any]) -> str:
        metadata = hit.get("metadata", {})
        paper_id = str(metadata.get("paper_id") or metadata.get("benchmark_doc_id") or "")
        section_id = str(
            metadata.get("benchmark_section_id")
            or metadata.get("parent_id")
            or metadata.get("section_path")
            or metadata.get("chunk_id")
            or hash(hit.get("text", ""))
        )
        return f"{paper_id}::{section_id}"

    @classmethod
    def _aggregate_sections(
        cls, query_text: str, hits: list[dict[str, Any]], support_chunks: int = 3
    ) -> list[dict[str, Any]]:
        """Collapse chunk candidates into section-level results with multi-chunk evidence."""

        query_tokens = set(_tokenize(query_text))
        grouped: dict[str, list[dict[str, Any]]] = {}
        for hit in hits:
            grouped.setdefault(cls._section_key(hit), []).append(hit)

        aggregated: list[dict[str, Any]] = []
        for section_hits in grouped.values():
            ordered = sorted(section_hits, key=lambda hit: float(hit.get("rerank_score") or 0.0), reverse=True)
            support = ordered[: max(support_chunks, 1)]
            rerank_values = [float(hit.get("rerank_score") or 0.0) for hit in support]
            vector_values = [float(hit.get("vector_score_normalized") or 0.0) for hit in support]
            keyword_values = [float(hit.get("keyword_score_normalized") or 0.0) for hit in support]
            evidence_tokens: set[str] = set()
            for hit in support:
                metadata = hit.get("metadata", {})
                evidence_tokens.update(_tokenize(str(hit.get("text") or "")))
                evidence_tokens.update(_tokenize(str(metadata.get("section_path") or "")))
            coverage = len(query_tokens & evidence_tokens) / max(len(query_tokens), 1)
            mean_rerank = sum(rerank_values) / len(rerank_values)
            mean_vector = sum(vector_values) / len(vector_values)
            mean_keyword = sum(keyword_values) / len(keyword_values)
            section_score = (
                0.35 * max(rerank_values)
                + 0.15 * mean_rerank
                + 0.2 * mean_vector
                + 0.15 * mean_keyword
                + 0.15 * coverage
            )
            representative = dict(ordered[0])
            representative["section_score"] = section_score
            representative["section_chunk_count"] = len(section_hits)
            representative["section_query_coverage"] = coverage
            representative["section_supporting_chunk_ids"] = [
                str(hit.get("metadata", {}).get("chunk_id") or "") for hit in support
            ]
            # Preserve the most query-relevant evidence inside the section for
            # generation. The prompt builder applies the final character caps.
            supporting_texts: list[str] = []
            seen_texts: set[str] = set()
            for hit in support:
                supporting_text = str(hit.get("text") or "").strip()
                if supporting_text and supporting_text not in seen_texts:
                    supporting_texts.append(supporting_text)
                    seen_texts.add(supporting_text)
            representative["generation_text"] = "\n\n".join(supporting_texts)
            representative["aggregated_section"] = True
            aggregated.append(representative)

        aggregated.sort(key=lambda hit: float(hit.get("section_score") or 0.0), reverse=True)
        return aggregated

    @classmethod
    def _merge_section_candidates(
        cls,
        query_text: str,
        chunk_sections: list[dict[str, Any]],
        indexed_sections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Fuse chunk aggregation with independent section-centroid retrieval."""

        merged = {cls._section_key(hit): hit for hit in chunk_sections}
        query_tokens = set(_tokenize(query_text))
        for indexed_hit in indexed_sections:
            key = cls._section_key(indexed_hit)
            indexed_score = float(indexed_hit.get("rerank_score") or 0.0)
            vector_score = float(indexed_hit.get("vector_score_normalized") or 0.0)
            current = merged.get(key)
            if current is not None:
                current["section_index_score"] = indexed_score
                current["section_index_vector_score"] = vector_score
                current["section_document"] = indexed_hit.get("section_document", "")
                current["section_score"] = 0.75 * float(current.get("section_score") or 0.0) + 0.25 * indexed_score
                current["retrieval_source"] = "+".join(
                    sorted(set(str(current.get("retrieval_source") or "").split("+")) | {"section_vector"})
                ).strip("+")
                continue

            metadata = indexed_hit.get("metadata", {})
            evidence_tokens = set(_tokenize(str(indexed_hit.get("text") or "")))
            evidence_tokens.update(_tokenize(str(metadata.get("section_path") or "")))
            coverage = len(query_tokens & evidence_tokens) / max(len(query_tokens), 1)
            candidate = dict(indexed_hit)
            candidate["section_index_score"] = indexed_score
            candidate["section_index_vector_score"] = vector_score
            candidate["section_score"] = 0.5 * indexed_score + 0.35 * vector_score + 0.15 * coverage
            candidate["section_query_coverage"] = coverage
            candidate["section_chunk_count"] = int(metadata.get("section_chunk_count") or 1)
            candidate["section_supporting_chunk_ids"] = []
            candidate["aggregated_section"] = True
            merged[key] = candidate

        results = list(merged.values())
        results.sort(key=lambda hit: float(hit.get("section_score") or 0.0), reverse=True)
        return results

    @staticmethod
    def _diversify_sections(hits: list[dict[str, Any]], max_per_section: int) -> list[dict[str, Any]]:
        """Prevent one high-scoring section from consuming every final slot."""

        if max_per_section <= 0:
            return hits
        selected: list[dict[str, Any]] = []
        deferred: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        for hit in hits:
            metadata = hit.get("metadata", {})
            section_key = "::".join(
                str(metadata.get(key) or "") for key in ("paper_id", "parent_id", "section_path")
            )
            if not section_key.strip(":") or counts[section_key] < max_per_section:
                selected.append(hit)
                counts[section_key] += 1
            else:
                deferred.append(hit)
        return selected + deferred

    def _normalize_query_result(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        hits: list[dict[str, Any]] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            hits.append(
                {
                    "text": document,
                    "metadata": metadata or {},
                    "distance": distance,
                }
            )
        return hits

    def _normalize_get_result(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        rows: list[dict[str, Any]] = []
        for row_id, document, metadata in zip(ids, documents, metadatas):
            rows.append(
                {
                    "id": row_id,
                    "text": document,
                    "metadata": metadata or {},
                }
            )
        return rows

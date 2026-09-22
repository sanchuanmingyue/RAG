"""Corpus-wide semantic classification built on persisted section vectors."""

from __future__ import annotations

import json
import re
from typing import Any
from uuid import uuid4

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


def requested_category_count(text: str) -> int | None:
    match = re.search(r"(?:分成|分为|划分为?|聚成)\s*(\d{1,2})\s*(?:类|组)", text)
    if not match:
        return None
    return min(max(int(match.group(1)), 2), 12)


def _json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


class CorpusAnalysisService:
    """Cluster every indexed paper by research-background semantics."""

    def __init__(self, vector_store: Any, llm_client: Any | None = None) -> None:
        self.vector_store = vector_store
        self.llm_client = llm_client

    def classify_by_background(
        self,
        request: str,
        *,
        paper_ids: set[str] | None = None,
        category_count: int | None = None,
    ) -> dict[str, Any]:
        profiles = self.vector_store.get_corpus_background_profiles()
        if paper_ids is not None:
            profiles = [profile for profile in profiles if profile["paper_id"] in paper_ids]
        if len(profiles) < 2:
            raise ValueError("至少需要两篇已建立章节向量的论文才能进行全库分类。")

        matrix = np.asarray([profile["embedding"] for profile in profiles], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-12)
        requested = requested_category_count(request) or category_count
        if requested is not None:
            requested = min(max(requested, 2), max(len(profiles) - 1, 2))
        category_count, labels, silhouette = self._cluster(matrix, requested)
        keywords = self._cluster_keywords(profiles, labels, category_count)
        representatives = self._representatives(profiles, matrix, labels, category_count)
        names = self._name_categories(keywords, representatives)

        categories = []
        for cluster_id in range(category_count):
            members = [
                profile for profile, label in zip(profiles, labels.tolist()) if label == cluster_id
            ]
            named = names.get(cluster_id) or {}
            categories.append(
                {
                    "cluster_id": cluster_id,
                    "label": named.get("label") or self._fallback_label(keywords.get(cluster_id, [])),
                    "description": named.get("description") or "基于研究背景章节的语义相似度归类。",
                    "keywords": keywords.get(cluster_id, []),
                    "count": len(members),
                    "papers": [
                        {"paper_id": item["paper_id"], "file_name": item["file_name"]}
                        for item in members
                    ],
                    "representative_papers": [item["file_name"] for item in representatives[cluster_id]],
                }
            )
        categories.sort(key=lambda item: (-item["count"], item["label"]))
        for index, category in enumerate(categories, start=1):
            category["rank"] = index
        return {
            "artifact_id": f"corpus_cls_{uuid4().hex[:10]}",
            "analysis_type": "research_background_classification",
            "paper_count": len(profiles),
            "category_count": category_count,
            "silhouette_score": round(float(silhouette), 4) if silhouette is not None else None,
            "method": "persisted section embeddings + cosine-normalized KMeans + one-pass label generation",
            "categories": categories,
        }

    @staticmethod
    def _cluster(matrix: np.ndarray, requested: int | None) -> tuple[int, np.ndarray, float | None]:
        sample_count = len(matrix)
        if requested is not None:
            category_count = min(requested, sample_count)
            model = KMeans(n_clusters=category_count, random_state=42, n_init=10)
            labels = model.fit_predict(matrix)
            score = silhouette_score(matrix, labels, metric="cosine") if category_count < sample_count else None
            return category_count, labels, score

        if sample_count <= 3:
            category_count = 2
            labels = KMeans(n_clusters=category_count, random_state=42, n_init=10).fit_predict(matrix)
            return category_count, labels, None

        candidates = range(3, min(8, sample_count - 1) + 1)
        best: tuple[float, int, np.ndarray] | None = None
        for category_count in candidates:
            labels = KMeans(n_clusters=category_count, random_state=42, n_init=10).fit_predict(matrix)
            if len(set(labels.tolist())) < 2:
                continue
            score = float(silhouette_score(matrix, labels, metric="cosine"))
            if best is None or score > best[0]:
                best = (score, category_count, labels)
        if best is None:
            labels = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(matrix)
            return 2, labels, None
        return best[1], best[2], best[0]

    @staticmethod
    def _cluster_keywords(
        profiles: list[dict[str, Any]], labels: np.ndarray, category_count: int
    ) -> dict[int, list[str]]:
        texts = [profile["text"] for profile in profiles]
        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                max_features=2500,
                ngram_range=(1, 2),
                min_df=2 if len(texts) >= 10 else 1,
            )
            values = vectorizer.fit_transform(texts)
            features = np.asarray(vectorizer.get_feature_names_out())
        except ValueError:
            return {cluster_id: [] for cluster_id in range(category_count)}

        output: dict[int, list[str]] = {}
        for cluster_id in range(category_count):
            indices = np.where(labels == cluster_id)[0]
            if not len(indices):
                output[cluster_id] = []
                continue
            weights = np.asarray(values[indices].mean(axis=0)).ravel()
            top_indices = weights.argsort()[::-1]
            terms = []
            for index in top_indices:
                term = str(features[index])
                if weights[index] <= 0:
                    break
                if any(term in existing or existing in term for existing in terms):
                    continue
                terms.append(term)
                if len(terms) == 6:
                    break
            output[cluster_id] = terms
        return output

    @staticmethod
    def _representatives(
        profiles: list[dict[str, Any]],
        matrix: np.ndarray,
        labels: np.ndarray,
        category_count: int,
    ) -> dict[int, list[dict[str, Any]]]:
        output: dict[int, list[dict[str, Any]]] = {}
        for cluster_id in range(category_count):
            indices = np.where(labels == cluster_id)[0]
            centroid = matrix[indices].mean(axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
            similarities = matrix[indices] @ centroid
            ranked = indices[np.argsort(similarities)[::-1][:3]]
            output[cluster_id] = [profiles[int(index)] for index in ranked]
        return output

    def _name_categories(
        self,
        keywords: dict[int, list[str]],
        representatives: dict[int, list[dict[str, Any]]],
    ) -> dict[int, dict[str, str]]:
        if self.llm_client is None:
            return {}
        clusters = []
        for cluster_id, samples in representatives.items():
            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "keywords": keywords.get(cluster_id, []),
                    "samples": [
                        {
                            "file": item["file_name"],
                            "background_excerpt": item["text"][:700],
                        }
                        for item in samples
                    ],
                }
            )
        prompt = (
            "你是科研文献分类助手。根据每个聚类的关键词和代表论文背景，为每个 cluster_id "
            "生成一个简洁、互相区分的中文研究背景类别名，以及一句不超过40字的描述。"
            "不得虚构样本之外的领域。只输出 JSON 对象，格式："
            '{"0":{"label":"类别名","description":"描述"}}。\n数据：'
            + json.dumps(clusters, ensure_ascii=False)
        )
        try:
            raw = self.llm_client.chat(
                [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=900
            )
            payload = _json_object(raw) or {}
        except Exception:
            return {}
        names: dict[int, dict[str, str]] = {}
        for key, value in payload.items():
            try:
                cluster_id = int(key)
            except (TypeError, ValueError):
                continue
            if not isinstance(value, dict):
                continue
            label = str(value.get("label") or "").strip()[:40]
            description = str(value.get("description") or "").strip()[:100]
            if label:
                names[cluster_id] = {"label": label, "description": description}
        return names

    @staticmethod
    def _fallback_label(keywords: list[str]) -> str:
        selected = keywords[:3]
        return " / ".join(selected) if selected else "其他研究背景"


def format_corpus_classification(payload: dict[str, Any]) -> str:
    lines = [
        f"已基于研究背景对 **{payload.get('paper_count', 0)} 篇论文**完成分类，"
        f"共得到 **{payload.get('category_count', 0)} 个类别**。"
    ]
    score = payload.get("silhouette_score")
    if score is not None:
        lines.append(f"聚类轮廓系数：`{score:.4f}`（用于自动选择类别数量）。")
    for category in payload.get("categories") or []:
        lines.append(
            f"\n### {category['rank']}. {category['label']}（{category['count']} 篇）\n"
            f"{category['description']}\n\n"
            + "\n".join(f"- `{paper['file_name']}`" for paper in category["papers"])
        )
    lines.append("\n> 分类依据为各论文的 Abstract、Introduction 与 Related Work 章节向量；结果属于语义聚类，可继续人工调整类别名称或论文归属。")
    return "\n".join(lines)

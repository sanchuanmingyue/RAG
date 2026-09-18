"""Natural-language literature search backed by the IEEE Xplore Metadata API."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import re
from typing import Any

import httpx

from backend.config import settings


_SEARCH_PATTERNS = (
    r"(?:查找|搜索|检索|找|推荐|搜)(?:一下|一些|几篇|相关)?(?:论文|文献|文章)",
    r"(?:论文|文献)(?:检索|搜索|推荐)",
    r"IEEE\s*(?:论文|文献|paper)",
    r"(?:find|search|recommend|look\s+for).{0,20}(?:papers?|literature|articles?)",
    r"literature\s+search",
)


def is_literature_search_query(text: str) -> bool:
    """Return True only for explicit requests to discover external papers."""

    normalized = " ".join(text.strip().split())
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in _SEARCH_PATTERNS)


@dataclass(frozen=True)
class SearchPlan:
    querytext: str
    start_year: int | None = None
    end_year: int | None = None
    explanation: str = "按用户原始关键词检索"


def _extract_json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _extract_year_range(text: str) -> tuple[int | None, int | None]:
    current_year = datetime.now().year
    range_match = re.search(r"\b(19\d{2}|20\d{2})\s*(?:年)?\s*[-—~～到至]\s*(19\d{2}|20\d{2})", text)
    if range_match:
        first, second = int(range_match.group(1)), int(range_match.group(2))
        return min(first, second), max(first, second)
    after_match = re.search(r"\b(19\d{2}|20\d{2})\s*年?\s*(?:以后|以来|之后|及以后|or later|onwards|since)", text, re.I)
    if after_match:
        return int(after_match.group(1)), current_year
    before_match = re.search(r"\b(19\d{2}|20\d{2})\s*年?\s*(?:以前|之前|及以前|or earlier|before)", text, re.I)
    if before_match:
        return None, int(before_match.group(1))
    recent_match = re.search(r"(?:近|最近|last|past)\s*(\d{1,2})\s*(?:年|years?)", text, re.I)
    if recent_match:
        years = min(max(int(recent_match.group(1)), 1), 30)
        return current_year - years + 1, current_year
    years = [int(value) for value in re.findall(r"\b(?:19\d{2}|20\d{2})\b", text)]
    return (years[0], years[0]) if len(years) == 1 else (None, None)


def _fallback_keywords(text: str) -> str:
    cleaned = re.sub(
        r"(?:请|帮我|给我|一下|一些|几篇|查找|搜索|检索|寻找|找|推荐|相关的?|关于|论文|文献|文章|IEEE)",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\b(?:19\d{2}|20\d{2})\b|(?:年|以后|以来|之后|以前|之前|近|最近)\s*\d*\s*年?", " ", cleaned)
    return " ".join(cleaned.split()).strip(" ，。,.；;:") or " ".join(text.split())


class IEEEQueryPlanner:
    """Translate a natural-language request into a compact IEEE query."""

    def __init__(self, llm_client: Any | None = None) -> None:
        self.llm_client = llm_client

    def plan(self, request: str) -> SearchPlan:
        start_year, end_year = _extract_year_range(request)
        fallback = SearchPlan(
            querytext=_fallback_keywords(request),
            start_year=start_year,
            end_year=end_year,
            explanation="已提取主题关键词与年份约束",
        )
        if self.llm_client is None:
            return fallback

        prompt = f"""你是 IEEE Xplore 检索式规划器。把用户的中文或英文需求转换成简洁的英文主题检索式。
只输出一个 JSON 对象，格式：
{{"querytext":"英文关键词或 AND/OR 布尔表达式","start_year":2022,"end_year":2026,"explanation":"一句中文说明"}}
规则：不要添加用户未提及的方法或领域；年份未知时填 null；querytext 不要包含年份；不要输出 Markdown。
用户需求：{request}"""
        try:
            raw = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            payload = _extract_json_object(raw)
            if not payload:
                return fallback
            querytext = " ".join(str(payload.get("querytext") or "").split())[:500]
            if not querytext:
                return fallback
            # Year limits are accepted only when they were explicitly present
            # in the user's request; the LLM must not invent narrower scope.
            planned_start = start_year
            planned_end = end_year
            if planned_start and planned_end and planned_start > planned_end:
                planned_start, planned_end = planned_end, planned_start
            return SearchPlan(
                querytext=querytext,
                start_year=planned_start,
                end_year=planned_end,
                explanation=str(payload.get("explanation") or fallback.explanation)[:300],
            )
        except Exception:
            return fallback


def _valid_year(value: Any) -> int | None:
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    return year if 1900 <= year <= datetime.now().year + 1 else None


def _authors(article: dict[str, Any]) -> list[str]:
    value = article.get("authors") or []
    if isinstance(value, dict):
        value = value.get("authors") or []
    if not isinstance(value, list):
        return []
    names = []
    for item in value:
        if isinstance(item, dict):
            name = item.get("full_name") or item.get("name")
        else:
            name = item
        if name and str(name).strip() not in names:
            names.append(str(name).strip())
    return names


def _terms(querytext: str) -> set[str]:
    stop = {"and", "or", "not", "the", "for", "with", "using", "based", "of", "in", "on", "a", "an"}
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{1,}", querytext)
        if token.lower() not in stop
    }


class IEEEXploreClient:
    def __init__(self, *, config=settings, http_client: httpx.Client | None = None) -> None:
        self.config = config
        self.http_client = http_client

    def search(self, plan: SearchPlan, *, limit: int = 10) -> dict[str, Any]:
        if not self.config.ieee_is_ready:
            raise RuntimeError("IEEE_API_KEY 尚未配置，请先在 .env 中填写 IEEE Xplore API Key。")
        limit = min(max(int(limit), 1), 50)
        fetch_limit = min(max(limit * 5, 25), 200) if (plan.start_year or plan.end_year) else limit
        params = {
            "apikey": self.config.ieee_api_key,
            "format": "json",
            "querytext": plan.querytext,
            "max_records": fetch_limit,
            "start_record": 1,
        }
        owns_client = self.http_client is None
        client = self.http_client or httpx.Client(timeout=self.config.ieee_search_timeout_seconds)
        try:
            response = client.get(f"{self.config.ieee_api_base_url}/search/articles", params=params)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in {401, 403}:
                raise RuntimeError("IEEE API 鉴权失败，请检查 IEEE_API_KEY 是否有效。") from exc
            raise RuntimeError(f"IEEE Xplore API 请求失败（HTTP {status}）。") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError(f"IEEE Xplore API 请求失败：{exc}") from exc
        finally:
            if owns_client:
                client.close()

        raw_articles = payload.get("articles") or []
        if isinstance(raw_articles, dict):
            raw_articles = raw_articles.get("articles") or []
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for position, article in enumerate(raw_articles, start=1):
            if not isinstance(article, dict):
                continue
            normalized = self._normalize(article, plan, position)
            if normalized is None:
                continue
            year = normalized["year"]
            if plan.start_year and (not year or year < plan.start_year):
                continue
            if plan.end_year and (not year or year > plan.end_year):
                continue
            key = (normalized.get("doi") or normalized.get("article_number") or normalized["title"]).lower()
            if key in seen:
                continue
            seen.add(key)
            results.append(normalized)
        results.sort(key=lambda item: (-item["score"]["total"], item["api_position"]))
        results = results[:limit]
        for rank, item in enumerate(results, start=1):
            item["rank"] = rank
        return {
            "source": "IEEE Xplore Metadata API",
            "query_plan": asdict(plan),
            "total_records": int(payload.get("total_records") or len(raw_articles)),
            "returned": len(results),
            "results": results,
            "search_log": {
                "querytext": plan.querytext,
                "source_order": ["IEEE Xplore"],
                "searched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "fetched": len(raw_articles),
                "deduplication": "DOI → IEEE article number → normalized title",
            },
        }

    @staticmethod
    def _normalize(article: dict[str, Any], plan: SearchPlan, position: int) -> dict[str, Any] | None:
        title = " ".join(str(article.get("title") or article.get("article_title") or "").split())
        if not title:
            return None
        article_number = str(article.get("article_number") or "").strip()
        doi = str(article.get("doi") or "").strip()
        url = str(article.get("html_url") or article.get("abstract_url") or "").strip()
        if not url and article_number:
            url = f"https://ieeexplore.ieee.org/document/{article_number}"
        if not (doi or url):
            return None
        abstract = " ".join(str(article.get("abstract") or "").split())
        venue = " ".join(str(article.get("publication_title") or article.get("publisher") or "IEEE").split())
        year = _valid_year(article.get("publication_year"))
        author_names = _authors(article)
        score = IEEEXploreClient._score(plan.querytext, title, abstract, venue, year, doi, url, author_names, position)
        return {
            "rank": 0,
            "api_position": position,
            "title": title,
            "authors": author_names,
            "venue": venue,
            "year": year,
            "doi": doi or None,
            "article_number": article_number or None,
            "ieee_url": url or None,
            "abstract": abstract,
            "abstract_snippet": abstract[:420] + ("…" if len(abstract) > 420 else ""),
            "content_type": article.get("content_type"),
            "score": score,
            "rationale": f"IEEE API 原始排序第 {position}；主题匹配 {score['relevance']}/5，元数据完整度 {score['metadata']}/2。",
            "source": "IEEE Xplore",
        }

    @staticmethod
    def _score(
        querytext: str,
        title: str,
        abstract: str,
        venue: str,
        year: int | None,
        doi: str,
        url: str,
        authors: list[str],
        position: int,
    ) -> dict[str, float]:
        query_terms = _terms(querytext)
        title_terms = _terms(title)
        abstract_terms = _terms(abstract)
        overlap_title = len(query_terms & title_terms) / max(len(query_terms), 1)
        overlap_abstract = len(query_terms & abstract_terms) / max(len(query_terms), 1)
        api_signal = max(0.5, 2.0 - (position - 1) * 0.08)
        relevance = min(5.0, api_signal + overlap_title * 2.0 + overlap_abstract)
        venue_fit = 2.0 + (1.0 if any(term in venue.lower() for term in query_terms) else 0.0)
        age = datetime.now().year - year if year else 99
        recency = 2.0 if age <= 3 else 1.5 if age <= 5 else 1.0 if age <= 10 else 0.5 if year else 0.0
        metadata = sum((0.5 if value else 0.0) for value in (doi, url, abstract, authors))
        total = relevance + venue_fit + recency + metadata
        return {
            "relevance": round(relevance, 2),
            "venue_fit": round(venue_fit, 2),
            "recency": round(recency, 2),
            "metadata": round(metadata, 2),
            "total": round(total, 2),
        }


class LiteratureSearchService:
    def __init__(self, *, llm_client: Any | None = None, ieee_client: IEEEXploreClient | None = None) -> None:
        self.planner = IEEEQueryPlanner(llm_client)
        self.ieee_client = ieee_client or IEEEXploreClient()

    def search(self, request: str, *, limit: int | None = None) -> dict[str, Any]:
        clean_request = " ".join(request.split())
        if not clean_request:
            raise ValueError("文献检索需求不能为空。")
        plan = self.planner.plan(clean_request)
        result = self.ieee_client.search(plan, limit=limit or settings.ieee_search_default_limit)
        result["request"] = clean_request
        return result


def format_search_answer(payload: dict[str, Any]) -> str:
    """Compact answer used in Chat; cards retain the detailed metadata."""

    count = int(payload.get("returned") or 0)
    plan = payload.get("query_plan") or {}
    query = plan.get("querytext") or ""
    if not count:
        return f"IEEE Xplore 中暂未找到满足条件的可核验文献。实际检索式：`{query}`。可尝试放宽年份或减少限定词。"
    return f"已从 IEEE Xplore 找到 **{count}** 篇可核验文献，并按相关性、期刊/会议匹配度、时效性和元数据完整度排序。实际检索式：`{query}`。"
